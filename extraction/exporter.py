"""
JSON export module.

Serializes the complete MotionData to a well-structured JSON file
with controlled float precision and numpy type handling.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from extraction.data_types import MotionData


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 6)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def export_motion_json(
    motion_data: MotionData,
    output_path: str,
    include_raw: bool = True,
    include_face_landmarks: bool = False,
    indent: int = 2,
) -> str:
    """
    Export MotionData to a JSON file.

    Args:
        motion_data: Complete motion data from the pipeline.
        output_path: Path for the output JSON file.
        include_raw: Include raw/world landmarks.
        include_face_landmarks: Include all 478 face landmarks.
        indent: JSON indentation (use None for compact).

    Returns:
        Absolute path to the written file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = motion_data.to_dict(
        include_raw=include_raw,
        include_face_landmarks=include_face_landmarks,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=_NumpyEncoder, indent=indent, ensure_ascii=False)

    file_size = os.path.getsize(output_path)
    print(f"[exporter] Wrote {output_path} ({file_size / 1024:.1f} KB)")
    print(f"[exporter]   frames: {len(motion_data.frames)}")
    print(f"[exporter]   include_raw: {include_raw}")
    print(f"[exporter]   include_face_landmarks: {include_face_landmarks}")

    return os.path.abspath(output_path)


def export_summary_json(
    motion_data: MotionData,
    tracking_summary: dict,
    output_path: str,
) -> str:
    """
    Export a lightweight summary JSON alongside the full motion data.

    Includes video metadata, tracking statistics, and high-level stats.
    """
    summary_path = output_path.replace(".json", "_summary.json")
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)

    # Compute high-level statistics
    n = len(motion_data.frames)
    hand_frames_left = sum(1 for f in motion_data.frames if f.left_hand.detected)
    hand_frames_right = sum(1 for f in motion_data.frames if f.right_hand.detected)
    face_frames = sum(1 for f in motion_data.frames if f.face.detected)
    body_frames = sum(1 for f in motion_data.frames if f.body.detected)

    # Motion phase distribution
    phases: dict[str, int] = {}
    for f in motion_data.frames:
        if f.motion_phase:
            phases[f.motion_phase] = phases.get(f.motion_phase, 0) + 1

    summary = {
        "metadata": motion_data.metadata.to_dict(),
        "tracking": tracking_summary,
        "statistics": {
            "total_frames": n,
            "body_detected_frames": body_frames,
            "left_hand_detected_frames": hand_frames_left,
            "right_hand_detected_frames": hand_frames_right,
            "face_detected_frames": face_frames,
            "motion_phases": phases,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, cls=_NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"[exporter] Wrote summary: {summary_path}")
    return os.path.abspath(summary_path)
