from __future__ import annotations

from argparse import Namespace

from connectors import YuNetSFaceConnector
from face_core.base import FaceConnector


def available_connectors() -> tuple[str, ...]:
    return ("yunet-sface",)


def create_connector(args: Namespace) -> FaceConnector:
    """
    Connector factory.

    To add another stack later:
      1. Create connectors/my_connector.py
      2. Implement detect(), embed(), compare()
      3. Import/register it here.
    """

    if args.stack == "yunet-sface":
        return YuNetSFaceConnector(
            model_dir=args.model_dir,
            yunet_model=args.yunet_model,
            sface_model=args.sface_model,
            detect_only=args.detect_only,
            det_threshold=args.det_threshold,
        )

    raise ValueError(f"Unknown connector: {args.stack}")