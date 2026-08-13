"""
Motion feature extraction module.

Computes velocity and acceleration for key landmarks using actual
timestamps, and optionally detects temporal motion events (phases)
from velocity patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from extraction.data_types import FrameData, LandmarkPoint


@dataclass
class MotionFeatures:
    """Velocity and acceleration for a tracked point."""
    velocity: list[float] = field(default_factory=list)       # [vx, vy] normalised units/sec
    acceleration: list[float] = field(default_factory=list)   # [ax, ay] normalised units/sec²
    speed: float = 0.0                                        # magnitude of velocity


# Landmark names to track for motion features
TRACKED_BODY_LANDMARKS = [
    "left_wrist", "right_wrist",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
]

TRACKED_HAND_POINTS = ["palm_center"]
TRACKED_HEAD_POINTS = ["head_position"]


def compute_frame_velocities(
    frames: list[FrameData],
    fps: float,
) -> None:
    """
    Compute velocity and acceleration for wrists, hands, and head.
    Mutates FrameData in-place by setting velocity/acceleration fields.

    Uses central differences where possible, forward/backward at boundaries.
    """
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    n = len(frames)

    if n < 2:
        return

    # --- Wrist velocities (from body landmarks) ---
    _compute_body_landmark_motion(frames, dt)

    # --- Hand center velocities ---
    _compute_hand_motion(frames, dt)

    # --- Head velocity (optional, for completeness) ---
    # Head is extracted separately in head_extractor, no extra velocity needed here


def _compute_body_landmark_motion(frames: list[FrameData], dt: float) -> None:
    """Compute velocity/acceleration for body landmarks across frames."""
    n = len(frames)

    for lm_name in TRACKED_BODY_LANDMARKS:
        # Extract positions over time
        positions: list[Optional[np.ndarray]] = []
        for f in frames:
            pos = _get_body_landmark_position(f, lm_name)
            positions.append(pos)

        # Compute velocities
        velocities = _finite_difference(positions, dt)

        # Compute accelerations from velocities
        accelerations = _finite_difference(
            [v if v is not None else None for v in velocities], dt,
        )

        # Store back into frame data (only for wrists, store in hand data)
        if "wrist" in lm_name:
            side = "left" if "left" in lm_name else "right"
            for i, f in enumerate(frames):
                hand = f.left_hand if side == "left" else f.right_hand
                if hand.detected:
                    if velocities[i] is not None:
                        hand.velocity = velocities[i].tolist()
                    if accelerations[i] is not None:
                        hand.acceleration = accelerations[i].tolist()


def _compute_hand_motion(frames: list[FrameData], dt: float) -> None:
    """Compute velocity/acceleration for hand palm centers."""
    n = len(frames)

    for side in ["left", "right"]:
        positions: list[Optional[np.ndarray]] = []
        for f in frames:
            hand = f.left_hand if side == "left" else f.right_hand
            if hand.detected and hand.palm_center:
                positions.append(np.array(hand.palm_center[:2]))
            else:
                positions.append(None)

        velocities = _finite_difference(positions, dt)
        accelerations = _finite_difference(
            [v if v is not None else None for v in velocities], dt,
        )

        for i, f in enumerate(frames):
            hand = f.left_hand if side == "left" else f.right_hand
            if hand.detected:
                if not hand.velocity and velocities[i] is not None:
                    hand.velocity = velocities[i].tolist()
                if not hand.acceleration and accelerations[i] is not None:
                    hand.acceleration = accelerations[i].tolist()


def _get_body_landmark_position(
    frame: FrameData,
    landmark_name: str,
) -> Optional[np.ndarray]:
    """Get normalized position of a body landmark by name."""
    if not frame.body.detected:
        return None
    for lm in frame.body.landmarks:
        if lm.name == landmark_name:
            return np.array([lm.x, lm.y])
    return None


def _finite_difference(
    positions: list[Optional[np.ndarray]],
    dt: float,
) -> list[Optional[np.ndarray]]:
    """
    Compute first derivative using central differences.

    Handles None values by propagating None.
    """
    n = len(positions)
    result: list[Optional[np.ndarray]] = [None] * n

    for i in range(n):
        if i == 0:
            # Forward difference
            if positions[0] is not None and positions[1] is not None:
                result[0] = (positions[1] - positions[0]) / dt
        elif i == n - 1:
            # Backward difference
            if positions[n - 1] is not None and positions[n - 2] is not None:
                result[n - 1] = (positions[n - 1] - positions[n - 2]) / dt
        else:
            # Central difference
            if positions[i - 1] is not None and positions[i + 1] is not None:
                result[i] = (positions[i + 1] - positions[i - 1]) / (2.0 * dt)

    return result


# ---------------------------------------------------------------------------
# Temporal event detection
# ---------------------------------------------------------------------------

# Motion phase thresholds (adaptive per video)
_NEUTRAL_THRESHOLD = 0.02      # normalised units/sec
_MOVEMENT_THRESHOLD = 0.08     # normalised units/sec
_HOLD_DURATION_FRAMES = 3      # minimum hold duration


def detect_motion_phases(
    frames: list[FrameData],
    fps: float,
) -> None:
    """
    Detect temporal motion events based on velocity patterns.

    Phases: neutral, preparation, movement, hold, transition

    Mutates FrameData.motion_phase in-place.
    """
    n = len(frames)
    if n < 2:
        return

    # Compute aggregate speed per frame (max of both wrists)
    speeds: list[float] = []
    for f in frames:
        wrist_speeds = []
        for hand in [f.left_hand, f.right_hand]:
            if hand.detected and hand.velocity:
                speed = float(np.linalg.norm(hand.velocity))
                wrist_speeds.append(speed)
        speeds.append(max(wrist_speeds) if wrist_speeds else 0.0)

    # Adaptive thresholds based on video statistics
    if speeds:
        median_speed = float(np.median([s for s in speeds if s > 0] or [0.0]))
        neutral_thresh = max(_NEUTRAL_THRESHOLD, median_speed * 0.2)
        movement_thresh = max(_MOVEMENT_THRESHOLD, median_speed * 0.6)
    else:
        neutral_thresh = _NEUTRAL_THRESHOLD
        movement_thresh = _MOVEMENT_THRESHOLD

    # Classify each frame
    prev_phase = "neutral"
    hold_counter = 0

    for i, speed in enumerate(speeds):
        if speed < neutral_thresh:
            if prev_phase == "movement":
                hold_counter += 1
                if hold_counter >= _HOLD_DURATION_FRAMES:
                    phase = "hold"
                else:
                    phase = "movement"  # brief pause, still movement
            else:
                phase = "neutral"
                hold_counter = 0
        elif speed < movement_thresh:
            if prev_phase == "neutral":
                phase = "preparation"
            elif prev_phase == "hold":
                phase = "transition"
            else:
                phase = "movement"
            hold_counter = 0
        else:
            phase = "movement"
            hold_counter = 0

        frames[i].motion_phase = phase
        prev_phase = phase
