# Lane Analytics Pipeline — Technical Write-up

## 1. Problem Definition
Prof. Avijit Maji — 4 deliverables:
1. Lane markings detection
2. Ego position within lane @ 1 Hz → CSV
3. Lane change events → CSV
4. Roadside signage → CSV

## 2. Video Characteristics
- 720×576 coded, 25 fps, ~1503 s
- Rural Indian highway, daytime
- Fixed lens artifact (windshield reflection)
- Heavy truck traffic in first ~45 s
- Some stretches without lane markings

## 3. Pipeline Architecture
[block diagram]

## 4. Mathematical Details
### 4.1 Canny Edge Detection
### 4.2 Hough Transform + Slope Classification
### 4.3 Trapezoid ROI + Horizon Detection
### 4.4 Sliding-Window 2nd-Degree Polynomial Fit
### 4.5 Per-Side Temporal State Machine
### 4.6 Ego Position (camera-center reference)
### 4.7 Lane-Change Detection (rolling-window state machine)
### 4.8 Sign Detection — YOLOv8n fine-tuned on 6,750-image Indian sign dataset

## 5. Results
### 5.1 Lane Detection — 1503 s processed, OK/MISS distribution
### 5.2 Ego Position — 1387 OK samples out of 1503
### 5.3 Lane Changes — 12 events
### 5.4 Sign Detection — [metrics after training]

## 6. Failure Modes & Fixes
- Truck specular highlights → component height cap
- Sky vs. lane colour confusion → Canny edges primary signal
- Horizon detector failures → ROI fallback
- Weathered paint → v_min lowered to 80

## 7. Rejected Approaches
- Pure HSV-primary (rejected after empirical failure)
- Hough straight-line only (breaks on curves)
- YOLOv8-Seg retraining on Amrita dataset (needs 22k labeled images + GPU days)
- Bennett SegNet + CamVid (road region ≠ lane geometry; UK domain mismatch)

## 8. Limitations
- Not validated on night/low-light/wet conditions
- Indian-specific sign classes: uses SDI dataset (85 classes)
- No IMU/GPS fusion — pixel/meter only