from __future__ import annotations

import argparse

import cv2 as cv
import numpy as np

from face_core import FaceDetection, draw_detection
from face_core.registry import available_connectors, create_connector
from face_gallery import count_embeddings, load_gallery, save_gallery


def largest_detection(detections: list[FaceDetection]) -> FaceDetection | None:
    if not detections:
        return None

    return max(detections, key=lambda detection: detection.bbox[2] * detection.bbox[3])


def recognize_embedding(
    *,
    connector,
    embedding: np.ndarray,
    gallery: dict[str, list[np.ndarray]],
    threshold: float,
) -> tuple[str, float]:
    best_label = "Unknown"
    best_score = -1.0

    for label, stored_embeddings in gallery.items():
        for stored_embedding in stored_embeddings:
            score = connector.compare(embedding, stored_embedding)

            if score > best_score:
                best_score = score
                best_label = label

    if best_score >= threshold:
        return best_label, best_score

    return "Unknown", best_score


def resize_for_speed(frame: np.ndarray, width: int | None) -> np.ndarray:
    if width is None or frame.shape[1] <= width:
        return frame

    scale = width / frame.shape[1]
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height))


def parse_camera(value: str):
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stack",
        default="yunet-sface",
        choices=available_connectors(),
        help="Face detection/recognition connector to use.",
    )

    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index or video path.",
    )

    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only run face detection, without recognition.",
    )

    parser.add_argument(
        "--enroll",
        help="Name to enroll, e.g. --enroll Tomiwa",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold. If omitted, the connector default is used.",
    )

    parser.add_argument(
        "--det-threshold",
        type=float,
        default=0.9,
        help="Face detection confidence threshold.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Resize video width for speed. Use 0 to keep original size.",
    )

    parser.add_argument(
        "--gallery",
        default="gallery.json",
        help="Path to saved face embeddings.",
    )

    parser.add_argument(
        "--allow-gallery-stack-mismatch",
        action="store_true",
        help="Allow loading a gallery created by a different connector. Usually avoid this.",
    )

    # Connector/model options. Future connectors can ignore these if they do not use them.
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Shared folder containing model files for the selected connector.",
    )

    parser.add_argument(
        "--yunet-model",
        default=None,
        help="Optional explicit path to the YuNet ONNX model.",
    )

    parser.add_argument(
        "--sface-model",
        default=None,
        help="Optional explicit path to the SFace ONNX model.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    connector = create_connector(args)
    threshold = args.threshold if args.threshold is not None else connector.default_threshold

    gallery = load_gallery(
        args.gallery,
        expected_stack=connector.name,
        allow_stack_mismatch=args.allow_gallery_stack_mismatch,
    )

    cap = cv.VideoCapture(parse_camera(args.camera))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video: {args.camera}")

    width = None if args.width == 0 else args.width

    if args.detect_only:
        print(f"Detection mode using connector: {connector.name}")
        print("Press q to quit.")
    elif args.enroll:
        print(f"Enrollment mode using connector: {connector.name}")
        print(f"Enrolling: {args.enroll}")
        print("Look at the camera and press s to save an embedding sample.")
        print("Take 3-5 samples with slightly different angles/lighting.")
        print("Press q to quit.")
    else:
        print(f"Recognition mode using connector: {connector.name}")
        print(f"Loaded {len(gallery)} identities and {count_embeddings(gallery)} embedding samples.")
        print(f"Recognition threshold: {threshold}")
        print("Press q to quit.")

    current_enroll_embedding = None

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = resize_for_speed(frame, width)
        detections = connector.detect(frame)

        current_enroll_embedding = None

        if args.detect_only:
            for detection in detections:
                draw_detection(frame, detection, f"Face {detection.score:.2f}")

        elif args.enroll:
            detection = largest_detection(detections)

            if detection is not None:
                current_enroll_embedding = connector.embed(frame, detection)
                draw_detection(frame, detection, f"Enroll: {args.enroll} | press s")

        else:
            for detection in detections:
                embedding = connector.embed(frame, detection)
                label, score = recognize_embedding(
                    connector=connector,
                    embedding=embedding,
                    gallery=gallery,
                    threshold=threshold,
                )

                if score >= 0:
                    draw_detection(frame, detection, f"{label} {score:.3f}")
                else:
                    draw_detection(frame, detection, label)

        cv.imshow("Face Detection + Recognition", frame)

        key = cv.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if args.enroll and key == ord("s") and current_enroll_embedding is not None:
            gallery.setdefault(args.enroll, []).append(current_enroll_embedding)
            save_gallery(
                args.gallery,
                gallery,
                stack_name=connector.name,
                metric=connector.metric,
            )
            count = len(gallery[args.enroll])
            print(f"Saved sample {count} for {args.enroll} into {args.gallery}")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()