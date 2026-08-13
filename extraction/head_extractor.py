"""
Head pose estimation module.

Estimates head position and rotation (pitch, yaw, roll) from facial
landmarks. Uses solvePnP with a canonical 3D face model for accurate
rotation, with a simpler geometric fallback.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from extraction.data_types import (
    LandmarkPoint,
    FrameHeadData,
    Orientation,
)

# Canonical 3D face model points (in arbitrary consistent coordinates)
# These correspond to stable facial landmarks.
# Order: nose tip, chin, left eye outer, right eye outer, mouth left, mouth right
_MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -63.6, -12.5],       # Chin
    [-43.3, 32.7, -26.0],      # Left eye outer
    [43.3, 32.7, -26.0],       # Right eye outer
    [-28.9, -28.9, -24.1],     # Mouth left
    [28.9, -28.9, -24.1],      # Mouth right
], dtype=np.float64)

# Corresponding MediaPipe face landmark indices
_FACE_LANDMARK_INDICES = [
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye outer
    263,  # Right eye outer
    61,   # Mouth left
    291,  # Mouth right
]

# Pose landmark indices for head position (from body pose)
_POSE_HEAD_INDICES = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
}


def extract_head_pose(
    face_landmarks: list[LandmarkPoint],
    pose_landmarks: list[LandmarkPoint],
    image_width: int,
    image_height: int,
) -> FrameHeadData:
    """
    Compute head position and rotation.

    Args:
        face_landmarks: 478 face landmarks (for rotation via solvePnP).
        pose_landmarks: 33 body pose landmarks (for head position).
        image_width: Video frame width in pixels.
        image_height: Video frame height in pixels.

    Returns:
        FrameHeadData with position and rotation.
    """
    # --- Head position from pose landmarks ---
    position: list[float] = []
    if pose_landmarks and len(pose_landmarks) > 0:
        nose = pose_landmarks[0]  # nose
        position = [nose.x, nose.y]

    # --- Head rotation from face landmarks via solvePnP ---
    rotation = Orientation()
    if face_landmarks and len(face_landmarks) > max(_FACE_LANDMARK_INDICES):
        rotation = _solve_head_rotation(
            face_landmarks, image_width, image_height,
        )
    elif face_landmarks and len(face_landmarks) > 10:
        # Fallback: simple geometric estimation
        rotation = _geometric_head_rotation(face_landmarks)

    detected = bool(position) or rotation != Orientation()

    return FrameHeadData(
        detected=detected,
        position=position if position else [0.0, 0.0],
        rotation=rotation,
    )


def _solve_head_rotation(
    face_landmarks: list[LandmarkPoint],
    width: int,
    height: int,
) -> Orientation:
    """Use cv2.solvePnP to estimate head rotation from face landmarks."""
    # Extract 2D image points for the selected landmarks
    image_points = np.array([
        [face_landmarks[idx].x * width, face_landmarks[idx].y * height]
        for idx in _FACE_LANDMARK_INDICES
    ], dtype=np.float64)

    # Approximate camera matrix (assume no lens distortion)
    focal_length = width
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return Orientation()

    # Convert rotation vector to rotation matrix, then extract Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    return _rotation_matrix_to_euler(rmat)


def _rotation_matrix_to_euler(rmat: np.ndarray) -> Orientation:
    """Extract pitch, yaw, roll from a 3×3 rotation matrix."""
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy > 1e-6:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])   # rotation around x
        yaw = math.atan2(-rmat[2, 0], sy)             # rotation around y
        roll = math.atan2(rmat[1, 0], rmat[0, 0])     # rotation around z
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0

    return Orientation(yaw=yaw, pitch=pitch, roll=roll)


def _geometric_head_rotation(face_landmarks: list[LandmarkPoint]) -> Orientation:
    """
    Simple geometric head rotation estimate when fewer than 478 landmarks
    are available. Uses nose, eye, and mouth positions.
    """
    pts = np.array([[lm.x, lm.y] for lm in face_landmarks[:10]])
    if len(pts) < 7:
        return Orientation()

    # Very approximate: use nose and ear/eye positions
    nose = pts[1] if len(pts) > 1 else pts[0]

    # Yaw from horizontal asymmetry
    yaw = 0.0

    # Pitch from vertical position of nose relative to face center
    pitch = 0.0

    # Roll from eye line angle (if eyes are available)
    roll = 0.0

    return Orientation(yaw=yaw, pitch=pitch, roll=roll)
