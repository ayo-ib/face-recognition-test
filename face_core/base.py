from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """
    Generic face detection result used by every connector.

    bbox: (x, y, width, height)
    landmarks: Optional Nx2 landmark array.
    raw: Optional connector-specific detection payload.
    """

    bbox: tuple[int, int, int, int]
    score: float
    landmarks: np.ndarray | None = None
    raw: object | None = None


class FaceConnector(Protocol):
    """
    Minimal interface every face connector must implement.

    The app owns the CLI/camera loop/gallery.
    A connector owns detection, embedding, and comparison.
    """

    name: str
    metric: str
    default_threshold: float

    @property
    def recognition_enabled(self) -> bool:
        ...

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        ...

    def embed(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        ...

    def compare(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        ...