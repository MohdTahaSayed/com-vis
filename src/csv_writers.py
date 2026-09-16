
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