"""
Face feature extraction module.

Extracts facial expression features from MediaPipe face landmarks
and ARKit-compatible blendshape coefficients. Computes derived
avatar-friendly features like eye openness, blink, mouth shape,
eyebrow position, and smile values.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from extraction.data_types import (
    LandmarkPoint,
    FrameFaceData,
)

# Key face landmark indices (MediaPipe 478-point mesh)
# These are stable across MediaPipe versions.
_FACE_INDICES = {
    # Eyes
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "right_eye_inner": 362,
    "right_eye_outer": 263,

    # Eyebrows
    "left_eyebrow_top": 105,
    "left_eyebrow_inner": 107,
    "right_eyebrow_top": 334,
    "right_eyebrow_inner": 336,

    # Mouth
    "mouth_top": 13,
    "mouth_bottom": 14,
    "mouth_left": 61,
    "mouth_right": 291,

    # Jaw / chin
    "chin": 152,

    # Nose
    "nose_tip": 1,

    # Forehead
    "forehead": 10,

    # Ears (for face width)
    "left_cheek": 234,
    "right_cheek": 454,
}


def extract_face_features(
    face_landmarks: list[LandmarkPoint],
    blendshapes: dict[str, float],
    include_landmarks: bool = False,
) -> FrameFaceData:
    """
    Compute derived facial features from landmarks and blendshapes.

    Uses blendshapes when available (more accurate), falls back to
    geometric computation from landmarks.
    """
    if not face_landmarks:
        return FrameFaceData(detected=False)

    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
    derived: dict[str, float] = {}

    # ---- Eye openness ----
    derived["eye_left_open"] = _eye_openness(pts, "left")
    derived["eye_right_open"] = _eye_openness(pts, "right")

    # ---- Blink values (from blendshapes or inverted openness) ----
    if "eyeBlinkLeft" in blendshapes:
        derived["blink_left"] = blendshapes["eyeBlinkLeft"]
        derived["blink_right"] = blendshapes.get("eyeBlinkRight", 0.0)
    else:
        derived["blink_left"] = max(0.0, 1.0 - derived["eye_left_open"])
        derived["blink_right"] = max(0.0, 1.0 - derived["eye_right_open"])

    # ---- Eyebrow position ----
    derived["eyebrow_left"] = _eyebrow_raise(pts, "left")
    derived["eyebrow_right"] = _eyebrow_raise(pts, "right")

    # ---- Mouth openness ----
    derived["mouth_open"] = _mouth_openness(pts)

    # ---- Mouth width ----
    derived["mouth_width"] = _mouth_width(pts)

    # ---- Jaw opening ----
    if "jawOpen" in blendshapes:
        derived["jaw_open"] = blendshapes["jawOpen"]
    else:
        derived["jaw_open"] = derived["mouth_open"]

    # ---- Smile ----
    if "mouthSmileLeft" in blendshapes and "mouthSmileRight" in blendshapes:
        derived["smile"] = (
            blendshapes["mouthSmileLeft"] + blendshapes["mouthSmileRight"]
        ) / 2.0
    else:
        derived["smile"] = 0.0

    # ---- Additional blendshape-derived values ----
    for bs_key in [
        "browInnerUp", "browDownLeft", "browDownRight",
        "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
        "mouthFunnel", "mouthPucker",
        "mouthLeft", "mouthRight",
        "noseSneerLeft", "noseSneerRight",
    ]:
        if bs_key in blendshapes:
            derived[bs_key] = blendshapes[bs_key]

    return FrameFaceData(
        detected=True,
        landmarks=face_landmarks if include_landmarks else None,
        blendshapes=blendshapes,
        derived=derived,
    )


def _eye_openness(pts: np.ndarray, side: str) -> float:
    """
    Compute eye openness as the ratio of vertical gap to horizontal width.

    Returns a value ~0 (closed) to ~1 (fully open).
    """
    if side == "left":
        top = pts[_FACE_INDICES["left_eye_top"]]
        bottom = pts[_FACE_INDICES["left_eye_bottom"]]
        inner = pts[_FACE_INDICES["left_eye_inner"]]
        outer = pts[_FACE_INDICES["left_eye_outer"]]
    else:
        top = pts[_FACE_INDICES["right_eye_top"]]
        bottom = pts[_FACE_INDICES["right_eye_bottom"]]
        inner = pts[_FACE_INDICES["right_eye_inner"]]
        outer = pts[_FACE_INDICES["right_eye_outer"]]

    vertical = float(np.linalg.norm(top - bottom))
    horizontal = float(np.linalg.norm(inner - outer))

    if horizontal < 1e-8:
        return 0.0

    # Typical EAR is ~0.2–0.4 when open; normalise to ~0–1
    ratio = vertical / horizontal
    return float(np.clip(ratio / 0.4, 0.0, 1.0))


def _eyebrow_raise(pts: np.ndarray, side: str) -> float:
    """
    Compute eyebrow raise as vertical distance from eyebrow to eye,
    normalised by face height (forehead to chin).
    """
    if side == "left":
        brow = pts[_FACE_INDICES["left_eyebrow_top"]]
        eye = pts[_FACE_INDICES["left_eye_top"]]
    else:
        brow = pts[_FACE_INDICES["right_eyebrow_top"]]
        eye = pts[_FACE_INDICES["right_eye_top"]]

    forehead = pts[_FACE_INDICES["forehead"]]
    chin = pts[_FACE_INDICES["chin"]]
    face_height = float(np.linalg.norm(forehead - chin))

    if face_height < 1e-8:
        return 0.0

    brow_eye_dist = abs(brow[1] - eye[1])
    return float(brow_eye_dist / face_height)


def _mouth_openness(pts: np.ndarray) -> float:
    """Mouth openness: vertical lip gap normalised by face height."""
    top = pts[_FACE_INDICES["mouth_top"]]
    bottom = pts[_FACE_INDICES["mouth_bottom"]]
    forehead = pts[_FACE_INDICES["forehead"]]
    chin = pts[_FACE_INDICES["chin"]]

    face_height = float(np.linalg.norm(forehead - chin))
    if face_height < 1e-8:
        return 0.0

    gap = float(np.linalg.norm(top - bottom))
    return float(np.clip(gap / (face_height * 0.15), 0.0, 1.0))


def _mouth_width(pts: np.ndarray) -> float:
    """Mouth width normalised by face width."""
    left = pts[_FACE_INDICES["mouth_left"]]
    right = pts[_FACE_INDICES["mouth_right"]]
    cheek_l = pts[_FACE_INDICES["left_cheek"]]
    cheek_r = pts[_FACE_INDICES["right_cheek"]]

    face_width = float(np.linalg.norm(cheek_l - cheek_r))
    if face_width < 1e-8:
        return 0.0

    width = float(np.linalg.norm(left - right))
    return float(np.clip(width / face_width, 0.0, 1.0))
