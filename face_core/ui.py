from __future__ import annotations

import cv2 as cv
import numpy as np

from .base import FaceDetection


def draw_detection(frame: np.ndarray, detection: FaceDetection, label: str) -> None:
    x, y, width, height = detection.bbox

    cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    if detection.landmarks is not None:
        for point in detection.landmarks.astype(np.int32):
            px, py = point
            cv.circle(frame, (int(px), int(py)), 2, (255, 0, 0), 2)

    cv.putText(
        frame,
        label,
        (x, max(20, y - 8)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )