from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

from face_core import FaceDetection


COSINE = getattr(cv, "FaceRecognizerSF_FR_COSINE", 0)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def resolve_model_path(model_dir: Path, explicit_path: str | None, filename: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return model_dir / filename


class YuNetSFaceConnector:
    """
    OpenCV YuNet + SFace connector.

    This is intentionally isolated from face_app.py so you can add or replace
    other model setups without touching the app loop.
    """

    name = "yunet-sface"
    metric = "cosine"
    default_threshold = 0.363

    def __init__(
        self,
        *,
        model_dir: str | Path = "models",
        yunet_model: str | None = None,
        sface_model: str | None = None,
        detect_only: bool = False,
        det_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        model_dir = Path(model_dir)

        self.yunet_path = resolve_model_path(
            model_dir,
            yunet_model,
            "face_detection_yunet_2023mar.onnx",
        )

        self.sface_path = (
            None
            if detect_only
            else resolve_model_path(
                model_dir,
                sface_model,
                "face_recognition_sface_2021dec.onnx",
            )
        )

        require_file(self.yunet_path)

        if self.sface_path is not None:
            require_file(self.sface_path)

        self.detector = cv.FaceDetectorYN.create(
            str(self.yunet_path),
            "",
            (320, 320),
            det_threshold,
            nms_threshold,
            top_k,
        )

        self.recognizer = (
            None
            if self.sface_path is None
            else cv.FaceRecognizerSF.create(str(self.sface_path), "")
        )

    @property
    def recognition_enabled(self) -> bool:
        return self.recognizer is not None

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        detections: list[FaceDetection] = []

        for face in faces:
            coords = face.astype(np.int32)
            x, y, w, h = coords[:4]

            landmarks = np.array(
                [
                    [face[4], face[5]],
                    [face[6], face[7]],
                    [face[8], face[9]],
                    [face[10], face[11]],
                    [face[12], face[13]],
                ],
                dtype=np.float32,
            )

            detections.append(
                FaceDetection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    score=float(face[14]),
                    landmarks=landmarks,
                    raw=face,
                )
            )

        return detections

    def embed(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        if self.recognizer is None:
            raise RuntimeError("Recognition is disabled for this connector instance.")

        if detection.raw is None:
            raise ValueError("YuNet+SFace requires the raw YuNet face row for alignment.")

        aligned = self.recognizer.alignCrop(frame, detection.raw)
        embedding = self.recognizer.feature(aligned)
        return embedding.copy()

    def compare(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        if self.recognizer is None:
            raise RuntimeError("Recognition is disabled for this connector instance.")

        return float(self.recognizer.match(embedding_a, embedding_b, COSINE))