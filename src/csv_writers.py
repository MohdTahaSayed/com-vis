"""
CSV writers for all pipeline outputs.

ego_position.csv columns (per Prof. Maji's spec):
    timestamp_s       - seconds since video start
    frame             - frame index
    offset_px         - ego-x minus lane-center-x, pixels; + = ego right of center
    lane_width_px     - detected lane width at eval row, pixels
    offset_normalized - offset_px / lane_width_px (unitless, ~[-0.5,+0.5])
    confidence        - 0..1 from state machine
    status            - OK | MISS

Rows are flushed to disk immediately so a killed/crashed run preserves
whatever was processed.
"""
from __future__ import annotations

import csv
import os


class CSVWriter:
    def __init__(self, path: str, header: list):
        self.path = path
        self.header = header
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fh = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fh)
        self.writer.writerow(header)
        self.fh.flush()

    def row(self, *values):
        self.writer.writerow(values)
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


class EgoPositionCSV(CSVWriter):
    def __init__(self, path: str):
        super().__init__(path, [
            "timestamp_s", "frame",
            "offset_px", "lane_width_px", "offset_normalized",
            "confidence", "status",
        ])


class LaneChangesCSV(CSVWriter):
    def __init__(self, path: str):
        super().__init__(path, [
            "timestamp_s", "frame", "direction", "magnitude",
        ])


class SignsCSV(CSVWriter):
    def __init__(self, path: str):
        super().__init__(path, [
            "timestamp_s", "frame",
            "class", "category",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "confidence",
        ])