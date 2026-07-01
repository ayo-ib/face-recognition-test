from __future__ import annotations

import json
from pathlib import Path

import numpy as np

Gallery = dict[str, list[np.ndarray]]


def load_gallery(
    path: str | Path,
    *,
    expected_stack: str,
    allow_stack_mismatch: bool = False,
) -> Gallery:
    """
    Load saved embeddings.

    Supports the new metadata format and the older first-script format.
    """

    gallery_path = Path(path)

    if not gallery_path.exists():
        return {}

    try:
        data = json.loads(gallery_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid gallery file: {gallery_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Invalid gallery file: {gallery_path}")

    if "people" in data:
        stored_stack = data.get("stack")

        if stored_stack and stored_stack != expected_stack and not allow_stack_mismatch:
            raise ValueError(
                f"Gallery was created for stack '{stored_stack}', but current stack is "
                f"'{expected_stack}'. Use a separate gallery file for each connector."
            )

        people = data.get("people", {})
    else:
        people = data

    gallery: Gallery = {}

    for label, vectors in people.items():
        gallery[label] = [
            np.array(vector, dtype=np.float32).reshape(1, -1)
            for vector in vectors
        ]

    return gallery


def save_gallery(
    path: str | Path,
    gallery: Gallery,
    *,
    stack_name: str,
    metric: str,
) -> None:
    gallery_path = Path(path)
    gallery_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "stack": stack_name,
        "metric": metric,
        "people": {
            label: [
                embedding.flatten().astype(float).tolist()
                for embedding in embeddings
            ]
            for label, embeddings in gallery.items()
        },
    }

    gallery_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_embeddings(gallery: Gallery) -> int:
    return sum(len(embeddings) for embeddings in gallery.values())