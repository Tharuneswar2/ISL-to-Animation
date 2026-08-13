"""
Hand feature extraction module.

Computes derived hand features from the 21 MediaPipe hand landmarks:
palm center, palm orientation (yaw/pitch/roll), finger joint angles,
finger curl, spread, openness, and hand size.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from extraction.data_types import (
    LandmarkPoint,
    WorldLandmarkPoint,
    FrameHandData,
    Orientation,
    FingerFeatures,
)

# Finger definitions: (name, [mcp, pip, dip, tip] landmark indices)
# Thumb is special: CMC=1, MCP=2, IP=3, TIP=4
FINGER_DEFS: dict[str, list[int]] = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# Adjacent finger pairs for spread calculation
ADJACENT_FINGERS: list[tuple[str, str]] = [
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
]


def extract_hand_features(
    landmarks: list[LandmarkPoint],
    world_landmarks: list[WorldLandmarkPoint],
) -> FrameHandData:
    """
    Compute all derived hand features from raw landmarks.

    Args:
        landmarks: 21 normalized hand landmarks.
        world_landmarks: 21 world (metre) hand landmarks.

    Returns:
        Populated FrameHandData.
    """
    if len(landmarks) < 21:
        return FrameHandData(detected=False)

    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Palm center: average of wrist(0), index_MCP(5), pinky_MCP(17)
    palm_center = pts[[0, 5, 17]].mean(axis=0).tolist()

    # Palm orientation
    orientation = _compute_palm_orientation(pts)

    # Finger features
    finger_features: dict[str, FingerFeatures] = {}
    for finger_name, indices in FINGER_DEFS.items():
        finger_features[finger_name] = _compute_finger_features(pts, indices, finger_name)

    # Finger spread between adjacent fingers
    for finger_a, finger_b in ADJACENT_FINGERS:
        spread = _compute_finger_spread(pts, finger_a, finger_b)
        finger_features[finger_a].spread = spread

    # Hand openness: average extension across all fingers
    openness = np.mean([ff.extension for ff in finger_features.values()])

    # Hand size: distance from wrist to middle fingertip
    hand_size = float(np.linalg.norm(pts[12] - pts[0]))

    # Local landmarks (relative to wrist)
    local_landmarks = _compute_local_landmarks(landmarks, pts)

    return FrameHandData(
        detected=True,
        landmarks=landmarks,
        world_landmarks=world_landmarks,
        local_landmarks=local_landmarks,
        palm_center=palm_center,
        orientation=orientation,
        finger_features=finger_features,
        openness=float(openness),
        hand_size=hand_size,
    )


def _compute_palm_orientation(pts: np.ndarray) -> Orientation:
    """
    Estimate palm orientation from hand geometry.

    Uses the cross product of wrist→index_MCP and wrist→pinky_MCP
    to get the palm normal, then extracts yaw/pitch/roll.
    """
    wrist = pts[0]
    index_mcp = pts[5]
    pinky_mcp = pts[17]

    v1 = index_mcp - wrist   # wrist → index_MCP
    v2 = pinky_mcp - wrist   # wrist → pinky_MCP

    # Palm normal (cross product)
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return Orientation()
    normal = normal / norm

    # Palm forward direction (average of v1 and v2, normalized)
    forward = (v1 + v2) / 2.0
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        return Orientation()
    forward = forward / forward_norm

    # Yaw: angle of forward direction projected onto XZ plane
    yaw = float(math.atan2(forward[0], forward[2])) if abs(forward[2]) > 1e-8 else 0.0

    # Pitch: angle of forward direction from horizontal
    pitch = float(math.asin(np.clip(-forward[1], -1.0, 1.0)))

    # Roll: angle of palm normal around the forward axis
    # Project normal onto the plane perpendicular to forward
    up = np.array([0.0, -1.0, 0.0])  # screen-up
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        roll = 0.0
    else:
        right = right / right_norm
        local_up = np.cross(right, forward)
        roll = float(math.atan2(np.dot(normal, right), np.dot(normal, local_up)))

    return Orientation(yaw=yaw, pitch=pitch, roll=roll)


def _compute_finger_features(
    pts: np.ndarray,
    indices: list[int],
    finger_name: str,
) -> FingerFeatures:
    """Compute joint angles and curl for a single finger."""
    # Joint angles at each bend point
    joint_angles: list[float] = []

    # For a 4-landmark finger: angle at joints [1] and [2]
    # (the joints between the bones)
    for j in range(1, len(indices) - 1):
        a = pts[indices[j - 1]]
        b = pts[indices[j]]
        c = pts[indices[j + 1]]
        angle = _angle_between_points(a, b, c)
        joint_angles.append(angle)

    # Also compute angle at MCP relative to wrist
    wrist = pts[0]
    mcp = pts[indices[0]]
    pip = pts[indices[1]]
    mcp_angle = _angle_between_points(wrist, mcp, pip)
    joint_angles.insert(0, mcp_angle)

    # Curl: ratio-based approach
    # Curl = 1 - (tip_to_mcp_distance / sum_of_bone_lengths)
    tip = pts[indices[-1]]
    base = pts[indices[0]]
    tip_to_base = float(np.linalg.norm(tip - base))

    total_bone_length = 0.0
    for j in range(len(indices) - 1):
        bone = pts[indices[j + 1]] - pts[indices[j]]
        total_bone_length += float(np.linalg.norm(bone))

    if total_bone_length > 1e-8:
        curl = 1.0 - min(1.0, tip_to_base / total_bone_length)
    else:
        curl = 0.0

    curl = max(0.0, min(1.0, curl))
    extension = 1.0 - curl

    return FingerFeatures(
        joint_angles=joint_angles,
        curl=curl,
        spread=0.0,  # filled in later by _compute_finger_spread
        extension=extension,
    )


def _compute_finger_spread(
    pts: np.ndarray,
    finger_a: str,
    finger_b: str,
) -> float:
    """Compute spread angle between two adjacent fingers at MCP level."""
    idx_a = FINGER_DEFS[finger_a]
    idx_b = FINGER_DEFS[finger_b]

    # Vectors from wrist to each MCP
    wrist = pts[0]
    mcp_a = pts[idx_a[0]]
    mcp_b = pts[idx_b[0]]

    va = mcp_a - wrist
    vb = mcp_b - wrist

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    cos_angle = np.dot(va, vb) / (norm_a * norm_b)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _angle_between_points(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """Compute the angle at point b formed by vectors b→a and b→c."""
    v1 = a - b
    v2 = c - b
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _compute_local_landmarks(
    landmarks: list[LandmarkPoint],
    pts: np.ndarray,
) -> list[LandmarkPoint]:
    """Compute hand landmarks relative to wrist (translation-invariant)."""
    wrist = pts[0]
    local: list[LandmarkPoint] = []
    for i, lm in enumerate(landmarks):
        rel = pts[i] - wrist
        local.append(LandmarkPoint(
            x=float(rel[0]),
            y=float(rel[1]),
            z=float(rel[2]),
            visibility=lm.visibility,
            name=lm.name,
        ))
    return local
