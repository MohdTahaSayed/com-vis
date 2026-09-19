from __future__ import annotations

import csv


class EgoPositionCSV:
    def __init__(self, path: str):
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp_s",
            "frame",
            "offset_px",
            "lane_width_px",
            "offset_normalized",
            "confidence",
            "status",
        ])

    def row(
        self,
        timestamp_s,
        frame,
        offset_px,
        lane_width_px,
        offset_normalized,
        confidence,
        status,
    ):
        self.writer.writerow([
            timestamp_s,
            frame,
            offset_px,
            lane_width_px,
            offset_normalized,
            confidence,
            status,
        ])

    def close(self):
        self.file.close()


class LaneChangesCSV:
    def __init__(self, path: str):
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp_s",
            "frame",
            "direction",
            "magnitude",
        ])

    def row(
        self,
        timestamp_s,
        frame,
        direction,
        magnitude,
    ):
        self.writer.writerow([
            timestamp_s,
            frame,
            direction,
            magnitude,
        ])

    def close(self):
        self.file.close()