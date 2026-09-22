# Lane Analytics — Video Analytics Model for Indian Highways

A classical-CV lane analytics pipeline for 720×576, 25-fps dashcam footage
of rural Indian highways. Implements the four required deliverables.

## Deliverables

| # | Deliverable | Output |
|---|---|---|
| 1 | Lane marking detection (on the fly) | `annotated.mp4` |
| 2 | Ego position within lane @ 1 Hz | `ego_position.csv` |
| 3 | Lane-change event timestamps | `lane_changes.csv` |
| 4 | Roadside signage + timestamp | `signs.csv` |

## Results on `VBOX0011_Trim.mp4`

| Deliverable | Result |
|---|---|
| Ego position @ 1 Hz | ~1,500 samples at 1 Hz |
| Lane changes | 12 events detected, 100% correct direction |
| Roadside signage | 0–1 confirmed (source-resolution limitation, see §5.4 of writeup) |

## Quick Start

```bash
pip install -r requirements.txt

# Step 1 — normalize (only needed if input isn't 720x576 @ 25 fps)
python scripts/normalize_video.py --input path/to/input.mp4 --outdir outputs_normalized

# Step 2 — run the full pipeline
python scripts/run_pipeline.py \
    --input outputs_normalized/input_720x576_25fps.mp4 \
    --outdir outputs/ \
    --debug-video