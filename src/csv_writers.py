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
            "lane_center_x",
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
        lane_center_x,
        confidence,
        status,
    ):
        self.writer.writerow([
            timestamp_s,
            frame,
            offset_px,
            lane_width_px,
            offset_normalized,
            lane_center_x,
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


class SignsCSV:
    def __init__(self, path: str):
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp_s",
            "frame",
            "class_name",
            "category",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
        ])

    def row(
        self,
        timestamp_s,
        frame,
        class_name,
        category,
        x1,
        y1,
        x2,
        y2,
        confidence,
    ):
        self.writer.writerow([
            timestamp_s,
            frame,
            class_name,
            category,
            x1,
            y1,
            x2,
            y2,
            confidence,
        ])

    def close(self):
        self.file.close()