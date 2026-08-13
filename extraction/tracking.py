"""
Temporal tracking module.

Maintains consistent left/right hand identity between frames,
handles temporary landmark loss, and ensures the frame sequence
is complete (no gaps in frame indices).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from extraction.data_types import LandmarkPoint


@dataclass
class TrackingState:
    """Tracks detection history for confidence-aware processing."""
    # Per-frame detection flags
    pose_detected: list[bool] = field(default_factory=list)
    left_hand_detected: list[bool] = field(default_factory=list)
    right_hand_detected: list[bool] = field(default_factory=list)
    face_detected: list[bool] = field(default_factory=list)

    # Gap tracking (consecutive missing frames)
    _left_hand_gap: int = 0
    _right_hand_gap: int = 0
    _pose_gap: int = 0
    _face_gap: int = 0

    def update(
        self,
        pose: bool,
        left_hand: bool,
        right_hand: bool,
        face: bool,
    ) -> None:
        """Record detection results for a new frame."""
        self.pose_detected.append(pose)
        self.left_hand_detected.append(left_hand)
        self.right_hand_detected.append(right_hand)
        self.face_detected.append(face)

        self._pose_gap = 0 if pose else self._pose_gap + 1
        self._left_hand_gap = 0 if left_hand else self._left_hand_gap + 1
        self._right_hand_gap = 0 if right_hand else self._right_hand_gap + 1
        self._face_gap = 0 if face else self._face_gap + 1

    @property
    def current_left_hand_gap(self) -> int:
        return self._left_hand_gap

    @property
    def current_right_hand_gap(self) -> int:
        return self._right_hand_gap

    def get_summary(self) -> dict:
        """Return summary statistics."""
        n = len(self.pose_detected)
        if n == 0:
            return {}
        return {
            "total_frames": n,
            "pose_detected_pct": sum(self.pose_detected) / n * 100,
            "left_hand_detected_pct": sum(self.left_hand_detected) / n * 100,
            "right_hand_detected_pct": sum(self.right_hand_detected) / n * 100,
            "face_detected_pct": sum(self.face_detected) / n * 100,
        }


def filter_low_confidence_landmarks(
    landmarks: list[LandmarkPoint],
    threshold: float = 0.5,
) -> tuple[list[LandmarkPoint], bool]:
    """
    Check if landmark visibility is above threshold.

    Returns:
        (landmarks, is_valid) — landmarks are kept but visibility
        is preserved for downstream use. is_valid indicates whether
        the average confidence is above threshold.
    """
    if not landmarks:
        return landmarks, False

    visibilities = [lm.visibility for lm in landmarks]
    avg_vis = np.mean(visibilities) if visibilities else 0.0

    return landmarks, float(avg_vis) >= threshold


def validate_frame_sequence(
    frame_indices: list[int],
    expected_count: int,
) -> list[int]:
    """
    Identify missing frame indices in the sequence.

    Returns a list of frame indices that are missing.
    """
    expected = set(range(expected_count))
    actual = set(frame_indices)
    return sorted(expected - actual)
