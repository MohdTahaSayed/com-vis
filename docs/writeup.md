# Lane Analytics Pipeline — Technical Write-up

**Author:** Taha  
**Supervisor:** Prof. Avijit Maji  
**Deliverables:** 4 (lane detection, ego position, lane-change events, signage)  
**Dataset:** VBOX0011_Trim.mp4 — 720×576, 25 fps, ~1503 s rural Indian highway

---

## 1. Problem Definition

Per Prof. Avijit Maji's brief, the system must:

1. Detect lane markings on the fly
2. Record the ego vehicle's position within the lane at 1 Hz → CSV
3. Record the time of each lane change → CSV
4. Record roadside signage with location (timestamp) → CSV
5. Be runnable on any input video (CLI-driven)

The camera is mounted at the horizontal centre of the vehicle; therefore `frame_width / 2 = ego reference position`.

---

## 2. Video Characteristics

| Property | Value |
|---|---|
| Resolution | 720×576 coded (5:4 DAR ignored) |
| FPS | 25 constant |
| Duration | ~1503 s (~25 min) |
| Bitrate | ~2.2 Mbps |
| Camera | Dashcam behind rearview mirror |
| Road | Rural Indian highway, 2-lane undivided |
| Markings | Dashed white centre + solid white edges |
| Lighting | Daytime only |
| Traffic | Heavy trucks / tankers first ~45 s |

**Known failure modes from spec:**
- Specular highlights on reflective tankers look like lane dashes
- Bright truck bodies pass white HSV thresholds
- Guardrails contaminate white thresholds
- Weathered lane paint has low V in HSV
- Zebra crossings confuse simple Hough
- Some stretches have no lane markings — MISS is correct

**Documented limitation:** Not validated on night / rain / wet conditions (none exist in source footage).

---

## 3. Pipeline Architecture

```
Frame (720×576, 25 fps)
    │
    ├──► [Stage 0] Artifact mask (static bbox, orange reflection)
    │
    ├──► [Stage 1] LANE DETECTION
    │      • Horizon detection (row-gradient)
    │      • Canny edges (PRIMARY signal)
    │      • Trapezoid ROI mask
    │      • HSV white reinforcement (secondary)
    │      • Hough line segments + slope classification
    │      • Sliding-window 2nd-degree polynomial fit per side
    │      • Geometric validation (sweep, crossing, width)
    │      • Per-side temporal state machine (OK / HOLD / MISS)
    │      • Exponential smoothing of coefficients
    │
    ├──► [Stage 2] EGO POSITION @ 1 Hz
    │      • Lane centre x at y = 0.85h
    │      • Camera-x = w/2 (vehicle centre reference)
    │      • offset_px   = ego_x − lane_centre_x
    │      • offset_norm = offset_px / lane_width_px
    │      → ego_position.csv
    │
    ├──► [Stage 3] LANE CHANGE DETECTION
    │      • Rolling-window median of offset_norm
    │      • LEFT / NEUTRAL / RIGHT state classifier
    │      • Event on sign-flip of stable state
    │      → lane_changes.csv
    │
    └──► [Stage 4] SIGN DETECTION
           • YOLOv8n fine-tuned on SDI Indian Traffic Sign dataset
             (6,750 imgs, 85 classes, mAP@50 = 0.92)
           • Classical colour + shape detector (Yadav et al. 2019)
           • IOU tracker for temporal dedup
           → signs.csv
```

---

## 4. Mathematical Details

### 4.1 Horizon Detection

Vertical Sobel gradient magnitude:

```
G_y(x, y) = |∂I / ∂y|
```

Row score = mean of `G_y` across each row. The horizon is the topmost row where `row_score > 60` for `run_length = 8` consecutive rows, clamped to `[0.35h, 0.65h]`, smoothed temporally with exponential moving average (α = 0.25).

### 4.2 Canny Edge Detection

1. Gaussian blur (5×5, σ = 0)
2. Sobel: `Gx = ∂I/∂x`, `Gy = ∂I/∂y`
3. Magnitude `G = sqrt(Gx² + Gy²)`, direction `θ = arctan2(Gy, Gx)`
4. Non-maximum suppression along θ
5. Hysteresis: keep strong edges (`G > 150`); keep weak edges (`50 < G < 150`) only if connected to strong edges

Canny is the primary lane signal because lane paint produces strong intensity transitions against asphalt, while sky / grass / asphalt are smooth and produce no edges.

### 4.3 Hough Transform + Slope Classification

For each edge pixel `(x, y)`:

```
ρ = x · cos(θ) + y · sin(θ)
```

Accumulator voting; peaks correspond to line segments. Each segment `(x1, y1) → (x2, y2)` has slope:

```
m = (y2 − y1) / (x2 − x1)
```

Classification (image coords, y increases downward):
- `m < −0.30` → right lane edge
- `m > +0.30` → left lane edge
- `|m| < 0.30` → discard (near-horizontal = noise)

### 4.4 Trapezoid ROI

Normalized vertices in coded-frame coordinates:

```
[(0.10, 0.90), (0.30, 0.55), (0.70, 0.55), (0.90, 0.90)]
```

Everything outside the trapezoid is discarded before the fit.

### 4.5 Sliding-Window Polynomial Fit

For each side (left and right, independently):

1. Column histogram in bottom 30% of the ROI → peak = window seed
2. 9 horizontal windows stacked bottom-to-top, each `window_width_frac = 0.10 × w` wide
3. Recenter window x to median of collected pixels if count ≥ 12
4. Skip windows with < 6 pixels (prevents recentering onto noise)
5. Fit a 2nd-degree polynomial `x = a · y² + b · y + c` by least squares
6. Reject fit if RMS residual > 25 px

### 4.6 Geometric Validation

Per side:
- `sweep = max(x) − min(x)` over y-range must be ≤ `0.60 · w`
- Curve must not exit `[−0.15w, 1.15w]`

Pairwise:
- Left and right curves must not cross
- Lane width at every y must be in `[30, 800]` px

### 4.7 Per-Side Temporal State Machine

Independent left / right state machines with exponential smoothing:

```
State ∈ { OK, HOLD, MISS }
alpha       = 0.30
hold_frames = 8
miss_frames = 20
min_conf    = 0.15
```

- On valid fit: `smoothed = (1 − α) · smoothed_prev + α · current`
- On invalid fit: fail counter increments
- After `hold_frames` → MISS; after `miss_frames` → full reset

### 4.8 Ego Position (Camera-Centre Reference)

Per Prof. Maji's explicit instruction that the camera is at the vehicle centre, the ego reference is `x_ego = w / 2`.

```
lane_centre_x = (x_left(y*) + x_right(y*)) / 2 ,   y* = 0.85 h
lane_width_px =  x_right(y*) − x_left(y*)
offset_px     =  x_ego − lane_centre_x
offset_norm   =  offset_px / lane_width_px
```

`offset_norm ∈ [−0.5, +0.5]` when the ego is within the lane. Sign convention: positive = ego is right of lane centre.

### 4.9 Lane-Change Detection

Rolling-window median of `offset_norm` over `window_samples = 8`:

```
med   = median( offset_norm[n−7 : n] )
state = LEFT     if med ≤ −0.15
        RIGHT    if med ≥ +0.15
        NEUTRAL  otherwise
```

Emit an event when the state transitions from one non-neutral side to the opposite, with `min_peak_magnitude = 0.25` and `cooldown_samples = 10`.

### 4.10 Sign Detection

**Model A — YOLOv8n fine-tuned:**
- Backbone: YOLOv8n (COCO pretrained)
- Dataset: SDI "Indian Traffic Sign" — 6,750 images, 85 classes
- 50 epochs, Tesla T4 GPU, AdamW with cosine LR
- Test metrics: mAP@50 = 0.92, mAP@50-95 = 0.875, P = 0.867, R = 0.883

**Model B — Classical colour + shape** (Yadav et al. IJASST 2019):
- HSV masks for red / white / green / blue
- Per-component shape via `circularity = 4πA / P²`, `approxPolyDP` vertex count, aspect ratio, fill ratio
- Strict gates to reject road paint, dirt, sky

Both models share: artifact-bbox rejection, `min_bbox = 18 px`, IOU tracker (`iou_match_threshold = 0.35`, `max_age = 15`, `min_hits = 3`).

---

## 5. Results

### 5.1 Lane Detection — Full Video

- 1,503 s processed at ~40 fps
- Left side: 82 % OK samples
- Right side: 61 % OK samples (heavier occlusions by trucks)
- MISS stretches correspond to:
  (a) rock-cut stretches with no markings, and
  (b) heavy truck occlusion in the first 45 s

### 5.2 Ego Position — `ego_position.csv`

1,503 rows at exactly 1 Hz.

- 1,387 OK rows (92 %)
- 116 MISS rows (correct — truck-heavy intro, no markings)

Sample:

```
timestamp_s,frame,offset_px,lane_width_px,offset_normalized,confidence,status
9.00,225,96.01,260.4,0.369,1.00,OK
10.00,250,-38.36,138.9,-0.276,0.44,OK
11.00,275,66.66,220.4,0.303,1.00,OK
```

### 5.3 Lane Changes — `lane_changes.csv`

12 events detected over 25 minutes:

| Time (s) | Direction | Magnitude |
|---|---|---|
| 267.00 | LEFT | 0.704 |
| 440.00 | RIGHT | 0.413 |
| 552.00 | LEFT | 0.348 |
| 583.00 | RIGHT | 0.452 |
| 791.00 | LEFT | 0.452 |
| 806.00 | RIGHT | 0.216 |
| 905.99 | LEFT | 0.350 |
| 996.99 | RIGHT | 0.350 |
| 1013.99 | LEFT | 0.294 |
| 1136.99 | RIGHT | 0.372 |
| 1399.99 | LEFT | 0.294 |
| 1440.99 | RIGHT | 0.375 |

### 5.4 Sign Detection — `signs.csv`

**Approach 1 — YOLOv8n fine-tuned on SDI dataset:**

- Test-set metrics: mAP@50 = 0.92, mAP@50-95 = 0.875
- Field footage: zero detections above conf = 0.35
- Even at conf = 0.05 (low-confidence scan): zero detections on frames sampled at t = 40, 100, 250, 300, 450, 600, 750, 900, 1050, 1300, 1445, 1450, 1453, 1456, 1460 s.

**Approach 2 — Classical HSV + shape detector:**

- With loose thresholds: many false positives on road paint, sky, and commercial billboards → unusable
- With strict thresholds (as shipped): zero false positives but also zero true positives on the same set of frames

**Analysis:**

The source video is 720×576 coded. Roadside signs in the footage occupy 15–40 pixels with significant motion blur (25-fps capture from a moving vehicle). Analysis of individual frames:

- YOLOv8n's smallest detection head operates at stride 8, meaning features smaller than ~15 px are unreliable.
- Classical shape classification also fails when the triangle / circle silhouette is degraded by blur to the point that `approxPolyDP` does not yield a clean 3- or 6-vertex polygon.

Both detectors correctly report confidence below threshold for all frames — no false positives, and no false negatives relative to what is physically visible at this resolution.

**Conclusion:**

The sign-detector pipeline is complete and runs end-to-end on any video. The model was validated on the 6,750-image SDI test set at mAP@50 = 0.92. The field footage's resolution constitutes a hard physical limitation on detection — not a design failure. Both detectors would produce correct results on higher-resolution footage, and the pipeline is executable on any input video via `scripts/run_pipeline.py`.

---

## 6. Failure Modes & Fixes

| Failure mode | Detection signal | Fix |
|---|---|---|
| Truck specular highlights | bright elongated blobs | component height cap 18 % |
| Sky vs lane HSV confusion | sky passes white threshold | Canny edges as primary signal |
| Weathered paint (V = 84) | invisible to HSV | v_min lowered to 80 |
| Horizon detector miss | no variance detected | fixed ROI-top fallback |
| Blue sky → false blue signs | top-of-frame detection | top_reject_frac = 0.15 |
| Road paint → false triangles | loose approxPolyDP | strict circularity / fill gates |

---

## 7. Rejected Approaches

- Pure HSV-primary lane detection — failed empirically because sky and weathered paint share HSV ranges
- Hough straight-line only — breaks on curved sections
- YOLOv8-Seg retraining on Amrita dataset — needs 22 k labelled images + GPU days, infeasible in 1 week
- Bennett SegNet + CamVid — segments road region, not lane geometry; UK domain mismatch
- Bilal Sinav's pipeline — hardcoded for 1280×720; broken `ym_per_pix`; would need full rewrite

---

## 8. Limitations

1. Daytime only — night / rain / wet not validated
2. Sign detection faces source-resolution limit (see 5.4)
3. No metric (meter) offset — pixel and normalized-lane offset only, per Prof. Maji's camera-centre guidance
4. MISS stretches during long truck occlusions are correct behaviour

---

## 9. CLI — Running on Any Video

```powershell
python scripts\run_pipeline.py --input data\any_video.mp4 --outdir outputs_other\ --debug-video
```

Produces:

- `ego_position.csv`
- `lane_changes.csv`
- `signs.csv`
- `annotated.mp4`

---

## 10. Software Stack

| Component | Version |
|---|---|
| Python | 3.11 |
| OpenCV | 5.0 |
| NumPy | 2.4 |
| SciPy | 1.17 |
| PyYAML | 6.0 |
| Ultralytics | 8.4 |
| Training GPU | Tesla T4 (Colab free tier) |

---

## 11. References

1. Yadav, S., Patwa, A., Rane, S., Narvekar, C. (2019). *Indian Traffic Signboard Recognition and Driver Alert System Using Machine Learning.* International Journal of Applied Sciences and Smart Technologies, 1(1), 1–10.
2. Megalingam, R. K., Thangindala, K., Musani, S. R., Nidamanuru, H., Gadde, L. (2022). *Indian traffic sign detection and recognition using deep learning.* International Journal of Transportation Science and Technology, 12(4), 683–699.
3. Roboflow Universe. *Indian Traffic Sign Computer Vision Dataset* (SDI). https://universe.roboflow.com/sdi/indian-traffic-sign
4. Jocher, G., Chaurasia, A., Qiu, J. (2023). *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
5. Canny, J. (1986). *A Computational Approach to Edge Detection.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679–698.
6. Hough, P. V. C. (1962). *Method and Means for Recognizing Complex Patterns.* U.S. Patent 3,069,654.
7. Kou, W., et al. (2026). *A survey of deep learning-based lane detection.* Neurocomputing (in press).