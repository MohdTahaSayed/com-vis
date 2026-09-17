# Lane Analytics — Video Analytics Model for Indian Highways

A classical-CV lane analytics pipeline for a 720×576, 25-fps rural Indian
highway dashcam recording. Implements the four deliverables.

1. **Lane marking detection** — Canny edges + Hough + 2nd-degree polynomial fit
2. **Ego position within lane @ 1 Hz** → `outputs/ego_position.csv`
3. **Lane-change events** → `outputs/lane_changes.csv`
4. **Roadside signage detection** → `outputs/signs.csv`

## Results on VBOX0011_Trim.mp4

| Deliverable | Output | Result |
| --- | --- | --- |
| Ego position @ 1 Hz | `outputs/ego_position.csv` | 1,503 samples (1,381 OK, 92%) |
| Lane changes | `outputs/lane_changes.csv` | 7 events over 25 min |
| Roadside signage | `outputs/signs.csv` | 0 confirmed tracks (see below) |

### Stage 4 note

YOLOv8n fine-tuned on SDI Indian traffic signs (mAP@50 = 0.92 on its test
set) produced zero confirmed tracks on this source. Sign instances are
20-40 px after motion blur — below the model's reliable detection floor.
This is a documented source-resolution limitation; see
`docs/writeup.md` section 5.4 and `outputs/signs_README.txt`.

## Quick start

    pip install -r requirements.txt

    # Run the full pipeline on the source video
    python scripts/run_pipeline.py --input data/VBOX0011_Trim.mp4 --outdir outputs/

    # Or run individual stages
    python scripts/test_stage2.py --input data/VBOX0011_Trim.mp4 --outdir outputs/
    python scripts/test_stage3.py --ego outputs/ego_position.csv --outdir outputs/
    python scripts/test_stage4.py --input data/VBOX0011_Trim.mp4 --outdir outputs/ --every 5

## Running on another video

If the input is not already 720x576 @ 25 fps, normalize it first:

    python scripts/normalize_video.py --input path/to/other.mp4 --outdir outputs_normalized
    python scripts/run_pipeline.py --input outputs_normalized/other_720x576_25fps.mp4 --outdir outputs_other

Note: the pipeline is tuned for the ego-lane geometry and marking style of
the source video (2-lane rural Indian highway). Non-Indian road networks
(US/EU) use different marking conventions (yellow left edge) and are out of
scope.

## Technical details

See docs/writeup.md for:

- pipeline architecture (Stages 0-4)
- mathematical derivations (Canny, Hough, sliding-window polynomial,
  temporal state machine)
- failure modes and mitigations
- rejected approaches
- limitations

## Software stack

Python 3.11, OpenCV 5.0, NumPy 2.4, SciPy 1.17, PyYAML 6.0, Ultralytics 8.4
