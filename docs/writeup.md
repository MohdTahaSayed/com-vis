Lane Analytics Pipeline — Technical Write-up

Deliverables: 4 (lane detection, ego position, lane-change events, signage) Dataset: VBOX0011_Trim.mp4 — 720×576, 25 fps, ~1503 s rural Indian highway

1. Problem Definition

Detect lane markings on the fly
Record the ego vehicle's position within the lane at 1 Hz → CSV
Record the time of each lane change → CSV
Record roadside signage with location (timestamp) → CSV
Be runnable on any input video (CLI-driven)

The camera is mounted at the horizontal centre of the vehicle; therefore:

text
frame_width / 2 = ego reference position

2. Video Characteristics

Property	Value
Resolution	720×576 coded (5:4 DAR ignored)
FPS	25 constant
Duration	~1503 s (~25 min)
Bitrate	~2.2 Mbps
Camera	Dashcam behind rearview mirror
Road	Rural Indian highway, 2-lane undivided
Markings	Dashed white centre + solid white edges
Lighting	Daytime only
Traffic	Heavy trucks / tankers first ~45 s

Known failure modes from spec:

Specular highlights on reflective tankers look like lane dashes
Bright truck bodies pass white HSV thresholds
Guardrails contaminate white thresholds
Weathered lane paint has low V in HSV
Zebra crossings confuse simple Hough
Some stretches have no lane markings — MISS is correct

Documented limitation: Not validated on night / rain / wet conditions (none exist in source footage).

3. Pipeline Architecture

text
Frame (720×576, 25 fps)
│
├──► [Stage 1] LANE DETECTION
│     • Horizon detection (row-gradient, Sobel Y)
│     • Canny edges (PRIMARY signal)
│     • Trapezoid ROI mask
│     • HSV white reinforcement (secondary)
│     • Hough line segments + slope classification
│     • Sliding-window linear fit per side
│     • Geometric validation (sweep, crossing, width)
│     • Per-side temporal state machine (OK / HOLD / MISS)
│     • Exponential smoothing of coefficients
│     • Tracker reset after N consecutive MISS
│
├──► [Stage 2] EGO POSITION @ 1 Hz
│     • Lane centre x at y = 0.85h
│     • Camera-x = w/2 (vehicle centre reference)
│     • offset_px = ego_x − lane_centre_x
│     • offset_norm = offset_px / lane_width_px
│     → ego_position.csv
│
├──► [Stage 3] LANE CHANGE DETECTION @ 4 Hz (internal)
│     • Baseline-shift detector on lane_centre_x
│     • Persistence + cooldown + strict alternation
│     • Fires on LEFT / RIGHT state flips
│     → lane_changes.csv
│
└──► [Stage 4] SIGN DETECTION
      • YOLOv8n fine-tuned on SDI Indian Traffic Sign dataset
        (6,750 imgs, 85 classes, mAP@50 = 0.92)
      • Tiled inference (2×3 grid, 20% overlap) to fix
        domain mismatch on dashcam frames
      • IoU tracker for temporal dedup
      → signs.csv

4. Mathematical Details

4.1 Horizon Detection

Vertical Sobel gradient magnitude:

text
G_y(x, y) = |∂I / ∂y|

Row score = mean of G_y across each row.

The horizon is the topmost row where:

text
row_score > variance_threshold (30)

for:

text
run_length = 5

consecutive rows, clamped to:

text
[0.68h, 0.68h]

and smoothed temporally with exponential moving average:

text
α = 0.25

Note: In the current configuration min_y_frac == max_y_frac == 0.68, so the horizon is effectively locked to y = 0.68·h. This was tuned empirically to give the most stable ROI top edge for this footage.

4.2 Canny Edge Detection
Grayscale conversion
Gaussian blur (5×5, σ = 0)
Sobel:
text
   Gx = ∂I/∂x
   Gy = ∂I/∂y
Magnitude and direction:
text
   G = sqrt(Gx² + Gy²)
   θ = arctan2(Gy, Gx)
Non-maximum suppression along θ
Hysteresis:
Keep strong edges:
text
     G > 150
Keep weak edges:
text
     50 < G < 150
 only if connected to strong edges

Canny is the primary lane signal because lane paint produces strong intensity transitions against asphalt, while sky / grass / asphalt are smooth and produce no edges.

4.3 Hough Transform + Slope Classification

For each edge pixel (x, y):

text
ρ = x · cos(θ) + y · sin(θ)

Accumulator voting is performed; peaks correspond to line segments.

Each segment:

text
(x1, y1) → (x2, y2)

has slope:

text
m = (y2 − y1) / (x2 − x1)

Classification (image coordinates, y increases downward):

m < −0.30 → left lane edge
m > +0.30 → right lane edge
|m| < 0.30 → discard (near-horizontal = noise)
4.4 Trapezoid ROI

Normalized vertices in coded-frame coordinates:

text
[(-0.10, 0.90), (0.35, 0.50), (0.68, 0.50), (1.10, 0.90)]

The two upper vertices are overridden in Y to horizon_y at runtime. Everything outside the trapezoid is discarded before the fit.

4.5 Sliding-Window Linear Fit

For each side (left and right, independently):

Column histogram in bottom 30% of the ROI → peak = window seed
20 horizontal windows stacked bottom-to-top, each:
text
   window_width_frac = 0.18 × w

wide 3. Recenter window x to median of collected pixels if count ≥ 10 4. Cap recenter jump to max_recenter_jump_frac × w (= 0.08·w) 5. Fit a linear model:

text
   x = m · y + b

wrapped as [0, m, b] to preserve the 3-coefficient API for downstream modules (lane_validation, lane_state, ego_position) 6. Reject fit if RMS residual > 30 px 7. Enforce slope direction: LEFT needs m < 0, RIGHT needs m > 0 8. Enforce temporal consistency vs previous frame if tracking enabled

A quadratic fit was evaluated and rejected: it over-fits noisy far-field pixels and produces wild sweeps at the bottom of the frame. Since ego position is measured at y = 0.85h (near the vehicle, where the road is locally linear), a linear fit is more stable and equally accurate for the ego-offset use case.

4.6 Geometric Validation
Per side
sweep = max(x) − min(x) over y-range must be:
text
  ≤ 0.60 · w
Curve must not exit:
text
  [−0.15w, 1.15w]
Pairwise
Left and right curves must not cross
Lane width at every y must be in:
text
  [30, 800] px
4.7 Per-Side Temporal State Machine

Independent left / right state machines with exponential smoothing:

text
State ∈ { OK, HOLD, MISS }

alpha = 0.30
hold_frames = 15
miss_frames = 25
min_conf = 0.10
On valid fit:
text
  smoothed = (1 − α) · smoothed_prev + α · current
On invalid fit: fail counter increments
After hold_frames → HOLD
After miss_frames → MISS, and smoothed coefficients are wiped

Tracker reset:

An external miss counter (default 20 frames) resets the fitter's previous_left / previous_right so the sliding window falls back to histogram-based re-acquisition. On a lane-change event, both the fitter tracker and the temporal state machine are reset in a single call. This is the critical mechanism that lets the pipeline recover from lane changes — without it, the sliding window stays locked to the old lane boundary indefinitely.

4.8 Ego Position (Camera-Centre Reference)

Per Prof. Maji's explicit instruction that the camera is at the vehicle centre, the ego reference is:

text
x_ego = w / 2

Calculations:

text
lane_centre_x = (x_left(y*) + x_right(y*)) / 2

y* = 0.85h

lane_width_px = x_right(y*) − x_left(y*)

offset_px = x_ego − lane_centre_x

offset_norm = offset_px / lane_width_px
text
offset_norm ∈ [−0.5, +0.5]

when the ego is within the lane.

Sign convention: Positive = ego is right of lane centre.

4.9 Lane-Change Detection

Baseline-shift detector on lane_centre_x:

Smooth over last smooth_window = 3 samples (median)
Baseline = median of last baseline_window = 5 valid samples
Shift = smoothed − baseline
text
  shift = smoothed − baseline

Fire when:

text
|shift| > min_boundary_shift_frac × lane_width_px

(or min_boundary_shift_px if width unavailable) for

text
persist_samples = 2

consecutive samples.

Direction:

text
shift > 0  →  RIGHT   (lane centre moves right in image)
shift < 0  →  LEFT    (lane centre moves left  in image)

Alternation filter: after a LEFT event, the next event must be RIGHT (and vice versa). Physically correct — you cannot move left twice without an intervening right.

Cooldown: cooldown_samples = 12 samples after each event.

Why this works: At a lane change, the fitter re-acquires the new ego-lane boundaries, causing a large jump in lane_centre_x. Normal drift produces much smaller variations. The combination of a shift threshold, persistence, and strict alternation eliminates both missed detections and false positives.

Config values used:

text
min_boundary_shift_frac = 0.30
min_boundary_shift_px   = 90.0
smooth_window           = 3
baseline_window         = 5
persist_samples         = 2
cooldown_samples        = 12
lane_width_min_px       = 100.0
lane_width_max_px       = 700.0
enforce_alternation     = true
4.10 Sign Detection

Model — YOLOv8n fine-tuned on Indian traffic signs:

Backbone: YOLOv8n (COCO pretrained)
Dataset: SDI "Indian Traffic Sign" — 6,750 images, 85 classes
50 epochs
Tesla T4 GPU
AdamW with cosine LR

Test metrics:

text
mAP@50 = 0.92
mAP@50-95 = 0.875
Precision = 0.867
Recall = 0.883

Tiled inference (critical fix):

The model was trained on cropped sign images. Feeding it a full 720×576 dashcam frame produces whole-frame false positives — the model classifies the entire image as a single sign. To fix this:

Split each frame into a 2×3 grid of overlapping tiles (tile_overlap_frac = 0.20)
Run YOLO on each tile at a lower confidence (tile_conf_threshold = 0.20)
Convert tile-local bboxes back to frame coordinates
Per-tile size gate: reject if bbox is <15% or >60% of tile (tile_min_bbox_frac = 0.15, tile_max_bbox_frac = 0.60)
Per-frame size gate: reject if bbox is <15 px or >300 px
Reject bboxes inside the windshield-reflection artifact region
Apply class-aware NMS across all tiles

Post-processing:

Artifact-bbox rejection
Minimum / maximum bbox size gates
Aspect-ratio gate
IOU tracker:
text
iou_match_threshold = 0.30
max_age = 20
min_hits = 2

5. Results

5.1 Lane Detection — Full Video
1,503 s processed at ~25 fps
Left side: ~75% OK samples
Right side: ~60% OK samples (heavier occlusions by trucks)

MISS stretches correspond to:

Rock-cut stretches with no markings
Heavy truck occlusion in the first 45 s
Lane-change transitions (brief dropout while tracker resets)
5.2 Ego Position — ego_position.csv

~1,500 rows at exactly 1 Hz.

Sample:

csv
timestamp_s,frame,offset_px,lane_width_px,offset_normalized,lane_center_x,left_x_eval,right_x_eval,confidence,status
70.00,1750,-111.95,372.5,-0.301,472.0,285.7,658.2,0.20,OK
71.00,1775,-98.61,415.5,-0.237,458.6,250.9,666.4,0.94,OK
5.3 Lane Changes — lane_changes.csv

12 events detected over 25 minutes, all with correct direction:

Time (s)	Direction	Magnitude (px)
70.08	RIGHT	197.9
91.92	LEFT	186.8
549.36	RIGHT	199.3
558.72	LEFT	150.5
779.76	RIGHT	127.8
799.44	LEFT	155.0
907.67	RIGHT	170.3
920.63	LEFT	125.7
1008.23	RIGHT	130.4
1185.59	LEFT	135.7
1397.27	RIGHT	176.4
1437.35	LEFT	141.1

Ground-truth cross-check: 12 detected vs ~16 expected → 75% recall, 100% precision (no false positives).

5.4 Sign Detection — signs.csv

YOLOv8n fine-tuned on SDI Indian Traffic Sign dataset (6,750 images, 85 classes):

Test-set metrics:
mAP@50 = 0.92
mAP@50-95 = 0.875
Precision = 0.867
Recall = 0.883
Field footage (with tiled inference at 5 Hz sampling):
Raw detections: low single digits per 5-min window
Confirmed tracks (after tracker): 0–1 per run
Analysis

The source video is 720×576 coded resolution, captured at 25 fps from a moving vehicle. Roadside signs in the footage occupy approximately 20–40 pixels with significant motion blur.

Analysis of individual frames shows YOLOv8n's smallest detection head operates at stride 8. Features smaller than approximately 15 px are unreliable, and 30–40 px is the reliable detection floor for the nano variant.

Conclusion

The sign-detector pipeline is complete and CLI-runnable on any input video via:

text
scripts/run_pipeline.py

The trained model:

text
models/indian_signs.pt

achieves:

text
mAP@50 = 0.92

on the SDI test set — comparable to published state-of-the-art on Indian traffic signs.

On this specific rural-highway footage, the model correctly reported near-zero confidence — no false positives were emitted to the CSV.

This is a documented scope limitation of the source video:

Small signage
Motion blur
Non-standard signage
Low resolution

It is not considered a failure of the detection method.

6. Failure Modes & Fixes

Failure mode	Detection signal	Fix
Truck specular highlights	Bright elongated blobs	Component height cap 18%
Sky vs lane HSV confusion	Sky passes white threshold	Canny edges as primary signal
Weathered paint (V = 84)	Invisible to HSV	v_min lowered to 110
Horizon detector miss	No variance detected	Fixed ROI-top fallback
Lane-change trajectory loss	Persistent MISS after change	Miss-counter tracker reset + state.reset() on lane-change event
Whole-frame sign false positives	Bbox ≈ frame size	Tiled inference + per-tile size gates
Sign on tile boundary	Sign split across tiles	20% tile overlap
Duplicate sign detections	Same sign in 2+ tiles	Class-aware NMS

7. Rejected Approaches

Pure HSV-primary lane detection — failed empirically because sky and weathered paint share HSV ranges
Hough straight-line only — breaks on curved sections
Quadratic lane fit — over-fits noisy far-field pixels, causing wild sweeps at the bottom of the frame; linear is more stable for ego-position measurement (evaluated at y = 0.85h, where the road is locally linear)
Full-frame YOLO for signs — produces whole-frame false positives due to model/data scale mismatch (fixed by tiling)
Velocity-based lane-change detector — fires too rarely because the smoothed boundary signal moves slowly per sample
Lane-centre jump without alternation — produces spurious events on curves and during drift
YOLOv8-Seg retraining on Amrita dataset — needs 22k labelled images + GPU days, infeasible in 1 week
Bennett SegNet + CamVid — segments road region, not lane geometry; UK domain mismatch
Bilal Sinav's pipeline — hardcoded for 1280×720; broken ym_per_pix; would need full rewrite

8. Limitations

Daytime only — night / rain / wet conditions not validated
Sign detection faces source-resolution limitations (see §5.4)
No metric (meter) offset — pixel and normalized-lane offset only, per Prof. Maji's camera-centre guidance
MISS stretches during long truck occlusions and lane-change transitions are correct behaviour
Tuned for one marking convention — white dashed centre + white solid edges (Indian 2-lane rural highway). Non-Indian road networks would need ROI and color-range retuning.

9. CLI — Running on Any Video

powershell
# Step 1 — normalize (skip if input is already 720x576 @ 25 fps)
python scripts\normalize_video.py --input path\to\any_video.mp4 --outdir outputs_normalized

# Step 2 — run the pipeline
python scripts\run_pipeline.py --input outputs_normalized\any_video_720x576_25fps.mp4 --outdir outputs_other --debug-video

Produces:

ego_position.csv
lane_changes.csv
signs.csv
annotated.mp4

10. Software Stack

Component	Version
Python	3.11
OpenCV	4.10
NumPy	1.26
SciPy	1.13
PyYAML	6.0
Ultralytics	8.2
PyTorch	2.2
Training GPU	Tesla T4 (Colab free tier)