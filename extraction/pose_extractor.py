"""
Pose extraction module.

Initialises the MediaPipe HolisticLandmarker and processes video frames
to extract body, hand, and face landmarks in a single pass.  Downloads
the model file automatically on first run.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from extraction.data_types import (
    LandmarkPoint,
    WorldLandmarkPoint,
)

# Model download URL and local cache path
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "holistic_landmarker.task")

# Pose landmark names (33 MediaPipe pose landmarks)
POSE_LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Hand landmark names (21 MediaPipe hand landmarks)
HAND_LANDMARK_NAMES: list[str] = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def ensure_model() -> str:
    """Download the holistic landmarker model if not present."""
    if os.path.isfile(_MODEL_PATH):
        return _MODEL_PATH

    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"[pose_extractor] Downloading model to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print(f"[pose_extractor] Model downloaded ({os.path.getsize(_MODEL_PATH) / 1e6:.1f} MB)")
    return _MODEL_PATH


class HolisticExtractor:
    """
    Wraps the MediaPipe HolisticLandmarker (Tasks API, VIDEO mode).

    Extracts pose, left/right hand, and face landmarks + blendshapes
    in a single inference pass per frame.
    """

    def __init__(
        self,
        min_pose_confidence: float = 0.5,
        min_hand_confidence: float = 0.5,
        min_face_confidence: float = 0.5,
    ) -> None:
        model_path = ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_pose_confidence,
            min_pose_landmarks_confidence=min_pose_confidence,
            min_hand_landmarks_confidence=min_hand_confidence,
            min_face_detection_confidence=min_face_confidence,
            min_face_landmarks_confidence=min_face_confidence,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_ms: int,
    ) -> "HolisticResult":
        """
        Run holistic detection on a single frame.

        Args:
            bgr_frame: OpenCV BGR image.
            timestamp_ms: Monotonically increasing timestamp in milliseconds.

        Returns:
            HolisticResult with parsed landmarks.
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return HolisticResult(result)

    def close(self) -> None:
        """Release the landmarker resources."""
        self._landmarker.close()

    def __enter__(self) -> "HolisticExtractor":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class HolisticResult:
    """Parsed wrapper around MediaPipe HolisticLandmarkerResult."""

    def __init__(self, raw_result: vision.HolisticLandmarkerResult) -> None:
        self._raw = raw_result

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------

    @property
    def has_pose(self) -> bool:
        return bool(self._raw.pose_landmarks)

    def get_pose_landmarks(self) -> list[LandmarkPoint]:
        if not self.has_pose:
            return []
        return [
            LandmarkPoint(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 0.0,
                name=POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"pose_{i}",
            )
            for i, lm in enumerate(self._raw.pose_landmarks)
        ]

    def get_pose_world_landmarks(self) -> list[WorldLandmarkPoint]:
        if not self._raw.pose_world_landmarks:
            return []
        return [
            WorldLandmarkPoint(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 0.0,
                name=POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"pose_{i}",
            )
            for i, lm in enumerate(self._raw.pose_world_landmarks)
        ]

    # ------------------------------------------------------------------
    # Hands
    # ------------------------------------------------------------------

    @property
    def has_left_hand(self) -> bool:
        return bool(self._raw.left_hand_landmarks)

    @property
    def has_right_hand(self) -> bool:
        return bool(self._raw.right_hand_landmarks)

    def get_hand_landmarks(self, side: str) -> list[LandmarkPoint]:
        """Get hand landmarks. side must be 'left' or 'right'."""
        raw = self._raw.left_hand_landmarks if side == "left" else self._raw.right_hand_landmarks
        if not raw:
            return []
        return [
            LandmarkPoint(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 1.0,
                name=HAND_LANDMARK_NAMES[i] if i < len(HAND_LANDMARK_NAMES) else f"hand_{i}",
            )
            for i, lm in enumerate(raw)
        ]

    def get_hand_world_landmarks(self, side: str) -> list[WorldLandmarkPoint]:
        raw = (
            self._raw.left_hand_world_landmarks
            if side == "left"
            else self._raw.right_hand_world_landmarks
        )
        if not raw:
            return []
        return [
            WorldLandmarkPoint(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 1.0,
                name=HAND_LANDMARK_NAMES[i] if i < len(HAND_LANDMARK_NAMES) else f"hand_{i}",
            )
            for i, lm in enumerate(raw)
        ]

    # ------------------------------------------------------------------
    # Face
    # ------------------------------------------------------------------

    @property
    def has_face(self) -> bool:
        return bool(self._raw.face_landmarks)

    def get_face_landmarks(self) -> list[LandmarkPoint]:
        if not self.has_face:
            return []
        return [
            LandmarkPoint(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=1.0,
                name=f"face_{i}",
            )
            for i, lm in enumerate(self._raw.face_landmarks)
        ]

    def get_face_blendshapes(self) -> dict[str, float]:
        if not self._raw.face_blendshapes:
            return {}
        return {
            cat.category_name: cat.score
            for cat in self._raw.face_blendshapes
        }
