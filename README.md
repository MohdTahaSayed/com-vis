# Lane Analytics — Video Analytics Model for Indian Highways

A complete video analytics pipeline for a 720×576, 25-fps rural Indian
highway dashcam recording. Implements the four deliverables specified
by Prof. Avijit Maji:

1. **Lane marking detection** — Canny edges + Hough + 2nd-degree polynomial fit
2. **Ego position within lane @ 1 Hz** → `ego_position.csv`
3. **Lane-change events** → `lane_changes.csv`
4. **Roadside signage detection** → `signs.csv`

## Quick start

```bash
pip install -r requirements.txt

# Run the full pipeline on any video
python scripts/run_pipeline.py --input data/VBOX0011_Trim.mp4 --outdir outputs/

# Or run individual stages
python scripts/test_stage2.py --input data/VBOX0011_Trim.mp4 --outdir outputs/
python scripts/test_stage3.py --ego outputs/ego_position.csv --outdir outputs/
python scripts/test_stage4.py --input data/VBOX0011_Trim.mp4 --outdir outputs/ --every 5