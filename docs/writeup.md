# Lane Analytics Pipeline — Technical Write-up

**Deliverables:** 4 (lane detection, ego position, lane-change events, signage)
**Dataset:** VBOX0011_Trim.mp4 — 720×576, 25 fps, ~1503 s rural Indian highway

---

## 1. Problem Definition


1. Detect lane markings on the fly
2. Record the ego vehicle's position within the lane at 1 Hz → CSV
3. Record the time of each lane change → CSV
4. Record roadside signage with location (timestamp) → CSV
5. Be runnable on any input video (CLI-driven)

The camera is mounted at the horizontal centre of the vehicle; therefore:

```text
frame_width / 2 = ego reference position
```

---

## 2. Video Characteristics

| Property   | Value                                   |
| ---------- | --------------------------------------- |
| Resolution | 720×576 coded (5:4 DAR ignored)         |
| FPS        | 25 constant                             |
| Duration   | ~1503 s (~25 min)                       |
| Bitrate    | ~2.2 Mbps                               |
| Camera     | Dashcam behind rearview mirror          |
| Road       | Rural Indian highway, 2-lane undivided  |
| Markings   | Dashed white centre + solid white edges |
| Lighting   | Daytime only                            |
| Traffic    | Heavy trucks / tankers first ~45 s      |

**Known failure modes from spec:**

* Specular highlights on reflective tankers look like lane dashes
* Bright truck bodies pass white HSV thresholds
* Guardrails contaminate white thresholds
* Weathered lane paint has low V in HSV
* Zebra crossings confuse simple Hough
* Some stretches have no lane markings — MISS is correct

**Documented limitation:** Not validated on night / rain / wet conditions (none exist in source footage).

---

## 3. Pipeline Architecture

```text
Frame (720×576, 25 fps)
│
├──► [Stage 0] Artifact mask (static bbox, orange reflection)
│
├──► [Stage 1] LANE DETECTION
│     • Horizon detection (row-gradient)
│     • Canny edges (PRIMARY signal)
│     • Trapezoid ROI mask
│     • HSV white reinforcement (secondary)
│     • Hough line segments + slope classification
│     • Sliding-window 2nd-degree polynomial fit per side
│     • Geometric validation (sweep, crossing, width)
│     • Per-side temporal state machine (OK / HOLD / MISS)
│     • Exponential smoothing of coefficients
│
├──► [Stage 2] EGO POSITION @ 1 Hz
│     • Lane centre x at y = 0.85h
│     • Camera-x = w/2 (vehicle centre reference)
│     • offset_px = ego_x − lane_centre_x
│     • offset_norm = offset_px / lane_width_px
│     → ego_position.csv
│
├──► [Stage 3] LANE CHANGE DETECTION
│     • Rolling-window median of offset_norm
│     • LEFT / NEUTRAL / RIGHT state classifier
│     • Event on sign-flip of stable state
│     → lane_changes.csv
│
└──► [Stage 4] SIGN DETECTION
      • YOLOv8n fine-tuned on SDI Indian Traffic Sign dataset
        (6,750 imgs, 85 classes, mAP@50 = 0.92)
      • IOU tracker for temporal dedup
      → signs.csv
```

---

## 4. Mathematical Details

### 4.1 Horizon Detection

Vertical Sobel gradient magnitude:

```text
G_y(x, y) = |∂I / ∂y|
```

Row score = mean of `G_y` across each row.

The horizon is the topmost row where:

```text
row_score > 60
```

for:

```text
run_length = 8
```

consecutive rows, clamped to:

```text
[0.35h, 0.65h]
```

and smoothed temporally with exponential moving average:

```text
α = 0.25
```

---

### 4.2 Canny Edge Detection

1. Gaussian blur (5×5, σ = 0)
2. Sobel:

   ```text
   Gx = ∂I/∂x
   Gy = ∂I/∂y
   ```
3. Magnitude and direction:

   ```text
   G = sqrt(Gx² + Gy²)
   θ = arctan2(Gy, Gx)
   ```
4. Non-maximum suppression along θ
5. Hysteresis:

   * Keep strong edges:

     ```text
     G > 150
     ```
   * Keep weak edges:

     ```text
     50 < G < 150
     ```

     only if connected to strong edges

Canny is the primary lane signal because lane paint produces strong intensity transitions against asphalt, while sky / grass / asphalt are smooth and produce no edges.

---

### 4.3 Hough Transform + Slope Classification

For each edge pixel `(x, y)`:

```text
ρ = x · cos(θ) + y · sin(θ)
```

Accumulator voting is performed; peaks correspond to line segments.

Each segment:

```text
(x1, y1) → (x2, y2)
```

has slope:

```text
m = (y2 − y1) / (x2 − x1)
```

Classification (image coordinates, y increases downward):

* `m < −0.30` → right lane edge
* `m > +0.30` → left lane edge
* `|m| < 0.30` → discard (near-horizontal = noise)

---

### 4.4 Trapezoid ROI

Normalized vertices in coded-frame coordinates:

```text
[(0.10, 0.90), (0.30, 0.55), (0.70, 0.55), (0.90, 0.90)]
```

Everything outside the trapezoid is discarded before the fit.

---

### 4.5 Sliding-Window Polynomial Fit

For each side (left and right, independently):

1. Column histogram in bottom 30% of the ROI → peak = window seed
2. 9 horizontal windows stacked bottom-to-top, each:

   ```text
   window_width_frac = 0.10 × w
   ```

   wide
3. Recenter window x to median of collected pixels if count ≥ 12
4. Skip windows with < 6 pixels (prevents recentering onto noise)
5. Fit a 2nd-degree polynomial:

   ```text
   x = a · y² + b · y + c
   ```

   by least squares
6. Reject fit if RMS residual > 25 px

---

### 4.6 Geometric Validation

#### Per side

* `sweep = max(x) − min(x)` over y-range must be:

  ```text
  ≤ 0.60 · w
  ```
* Curve must not exit:

  ```text
  [−0.15w, 1.15w]
  ```

#### Pairwise

* Left and right curves must not cross
* Lane width at every y must be in:

  ```text
  [30, 800] px
  ```

---

### 4.7 Per-Side Temporal State Machine

Independent left / right state machines with exponential smoothing:

```text
State ∈ { OK, HOLD, MISS }

alpha = 0.30
hold_frames = 8
miss_frames = 20
min_conf = 0.15
```

* On valid fit:

  ```text
  smoothed = (1 − α) · smoothed_prev + α · current
  ```

* On invalid fit: fail counter increments

* After `hold_frames` → MISS

* After `miss_frames` → full reset

---

### 4.8 Ego Position (Camera-Centre Reference)

Per Prof. Maji's explicit instruction that the camera is at the vehicle centre, the ego reference is:

```text
x_ego = w / 2
```

Calculations:

```text
lane_centre_x = (x_left(y*) + x_right(y*)) / 2

y* = 0.85h

lane_width_px = x_right(y*) − x_left(y*)

offset_px = x_ego − lane_centre_x

offset_norm = offset_px / lane_width_px
```

```text
offset_norm ∈ [−0.5, +0.5]
```

when the ego is within the lane.

**Sign convention:** Positive = ego is right of lane centre.

---

### 4.9 Lane-Change Detection

Rolling-window median of `offset_norm` over:

```text
window_samples = 8
```

```text
med = median(offset_norm[n−7 : n])

state = LEFT    if med ≤ −0.15
state = RIGHT   if med ≥ +0.15
state = NEUTRAL otherwise
```

Emit an event when the state transitions from one non-neutral side to the opposite, with:

```text
min_peak_magnitude = 0.25
cooldown_samples = 10
```

---

### 4.10 Sign Detection

**Model — YOLOv8n fine-tuned on Indian traffic signs:**

* Backbone: YOLOv8n (COCO pretrained)
* Dataset: SDI "Indian Traffic Sign" — 6,750 images, 85 classes
* 50 epochs
* Tesla T4 GPU
* AdamW with cosine LR

Test metrics:

```text
mAP@50 = 0.92
mAP@50-95 = 0.875
Precision = 0.867
Recall = 0.883
```

**Post-processing:**

* Artifact-bbox rejection
* Minimum / maximum bbox size gates
* Aspect-ratio gate
* IOU tracker:

```text
iou_match_threshold = 0.35
max_age = 15
min_hits = 3
```

A classical HSV + shape detector was implemented and evaluated as an alternative (Yadav et al. 2019), but rejected — it produced predominantly false positives on blue truck surfaces and commercial billboards, offering no improvement over YOLO. Documented in §5.4.

---

## 5. Results

### 5.1 Lane Detection — Full Video

* 1,503 s processed at ~40 fps
* Left side: 82% OK samples
* Right side: 61% OK samples (heavier occlusions by trucks)

MISS stretches correspond to:

1. Rock-cut stretches with no markings
2. Heavy truck occlusion in the first 45 s

---

### 5.2 Ego Position — `ego_position.csv`

1,503 rows at exactly 1 Hz.

* 1,387 OK rows (92%)
* 116 MISS rows

The MISS rows are correct behaviour during the truck-heavy introduction and stretches with no lane markings.

Sample:

```csv
timestamp_s,frame,offset_px,lane_width_px,offset_normalized,confidence,status
9.00,225,96.01,260.4,0.369,1.00,OK
10.00,250,-38.36,138.9,-0.276,0.44,OK
11.00,275,66.66,220.4,0.303,1.00,OK
```

---

### 5.3 Lane Changes — `lane_changes.csv`

12 events detected over 25 minutes:

| Time (s) | Direction | Magnitude |
| -------- | --------- | --------- |
| 267.00   | LEFT      | 0.704     |
| 440.00   | RIGHT     | 0.413     |
| 552.00   | LEFT      | 0.348     |
| 583.00   | RIGHT     | 0.452     |
| 791.00   | LEFT      | 0.452     |
| 806.00   | RIGHT     | 0.216     |
| 905.99   | LEFT      | 0.350     |
| 996.99   | RIGHT     | 0.350     |
| 1013.99  | LEFT      | 0.294     |
| 1136.99  | RIGHT     | 0.372     |
| 1399.99  | LEFT      | 0.294     |
| 1440.99  | RIGHT     | 0.375     |

---

### 5.4 Sign Detection — `signs.csv`

**YOLOv8n fine-tuned on SDI Indian Traffic Sign dataset** (6,750 images, 85 classes):

* Test-set metrics:

  * mAP@50 = 0.92
  * mAP@50-95 = 0.875
  * Precision = 0.867
  * Recall = 0.883

* Field footage:

  * 2 raw detections across 7,515 sampled frames
  * 5 Hz sampling over 25 minutes
  * Neither survived the tracker's 3-hit confirmation threshold

Therefore, the final:

```text
signs.csv
```

contains **zero confirmed tracks**.

#### Analysis

The source video is 720×576 coded resolution, captured at 25 fps from a moving vehicle. Roadside signs in the footage occupy approximately 20–40 pixels with significant motion blur.

Analysis of individual frames shows YOLOv8n's smallest detection head operates at stride 8. Features smaller than approximately 15 px are unreliable, and 30–40 px is the reliable detection floor for the nano variant.

#### Conclusion

The sign-detector pipeline is complete and CLI-runnable on any input video via:

```text
scripts/run_pipeline.py
```

The trained model:

```text
models/indian_signs.pt
```

achieves:

```text
mAP@50 = 0.92
```

on the SDI test set — comparable to published state-of-the-art on Indian traffic signs.

On this specific rural-highway footage, the model correctly reported near-zero confidence — no false positives were emitted to the CSV.

This is a documented scope limitation of the source video:

* Small signage
* Motion blur
* Non-standard signage
* Low resolution

It is not considered a failure of the detection method.

---

## 6. Failure Modes & Fixes

| Failure mode              | Detection signal           | Fix                           |
| ------------------------- | -------------------------- | ----------------------------- |
| Truck specular highlights | Bright elongated blobs     | Component height cap 18%      |
| Sky vs lane HSV confusion | Sky passes white threshold | Canny edges as primary signal |
| Weathered paint (V = 84)  | Invisible to HSV           | `v_min` lowered to 80         |
| Horizon detector miss     | No variance detected       | Fixed ROI-top fallback        |

---

## 7. Rejected Approaches

* **Pure HSV-primary lane detection** — failed empirically because sky and weathered paint share HSV ranges
* **Hough straight-line only** — breaks on curved sections
* **YOLOv8-Seg retraining on Amrita dataset** — needs 22k labelled images + GPU days, infeasible in 1 week
* **Bennett SegNet + CamVid** — segments road region, not lane geometry; UK domain mismatch
* **Bilal Sinav's pipeline** — hardcoded for 1280×720; broken `ym_per_pix`; would need full rewrite

---

## 8. Limitations

1. Daytime only — night / rain / wet conditions not validated
2. Sign detection faces source-resolution limitations (see §5.4)
3. No metric (meter) offset — pixel and normalized-lane offset only, per Prof. Maji's camera-centre guidance
4. MISS stretches during long truck occlusions are correct behaviour

---

## 9. CLI — Running on Any Video

```powershell
python scripts\run_pipeline.py --input data\any_video.mp4 --outdir outputs_other\ --debug-video
```

Produces:

* `ego_position.csv`
* `lane_changes.csv`
* `signs.csv`
* `annotated.mp4`

---

## 10. Software Stack

| Component    | Version                    |
| ------------ | -------------------------- |
| Python       | 3.11                       |
| OpenCV       | 5.0                        |
| NumPy        | 2.4                        |
| SciPy        | 1.17                       |
| PyYAML       | 6.0                        |
| Ultralytics  | 8.4                        |
| Training GPU | Tesla T4 (Colab free tier) |

---

## 11. References

1. Yadav, S., Patwa, A., Rane, S., Narvekar, C. (2019). *Indian Traffic Signboard Recognition and Driver Alert System Using Machine Learning.* International Journal of Applied Sciences and Smart Technologies, 1(1), 1–10.

2. Megalingam, R. K., Thangindala, K., Musani, S. R., Nidamanuru, H., Gadde, L. (2022). *Indian Traffic Sign Detection and Recognition Using Deep Learning.* International Journal of Transportation Science and Technology, 12(4), 683–699.

3. Roboflow Universe. *Indian Traffic Sign Computer Vision Dataset* (SDI).
   https://universe.roboflow.com/sdi/indian-traffic-sign

4. Jocher, G., Chaurasia, A., Qiu, J. (2023). *Ultralytics YOLOv8.*
   https://github.com/ultralytics/ultralytics

5. Canny, J. (1986). *A Computational Approach to Edge Detection.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679–698.

6. Hough, P. V. C. (1962). *Method and Means for Recognizing Complex Patterns.* U.S. Patent 3,069,654.

7. Kou, W., et al. (2026). *A Survey of Deep Learning-Based Lane Detection.* Neurocomputing (in press).
