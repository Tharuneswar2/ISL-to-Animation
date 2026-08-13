"""
Central data structures for the ISL motion extraction pipeline.

All modules share these typed dataclasses to ensure a consistent,
well-defined data flow from extraction through export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

@dataclass
class LandmarkPoint:
    """A single detected landmark in normalized image coordinates."""
    x: float              # normalized x (0..1)
    y: float              # normalized y (0..1)
    z: float = 0.0        # depth / relative z
    visibility: float = 0.0
    name: str = ""

    def to_dict(self) -> dict:
        d: dict = {"x": round(self.x, 6), "y": round(self.y, 6)}
        if self.z != 0.0:
            d["z"] = round(self.z, 6)
        d["visibility"] = round(self.visibility, 4)
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class WorldLandmarkPoint:
    """A landmark in real-world metres (from MediaPipe world landmarks)."""
    x: float
    y: float
    z: float
    visibility: float = 0.0
    name: str = ""

    def to_dict(self) -> dict:
        d: dict = {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "z": round(self.z, 6),
        }
        d["visibility"] = round(self.visibility, 4)
        if self.name:
            d["name"] = self.name
        return d


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """Metadata extracted from the input video file."""
    source_video: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float

    def to_dict(self) -> dict:
        return {
            "source_video": self.source_video,
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": round(self.duration, 4),
        }


# ---------------------------------------------------------------------------
# Orientation / rotation
# ---------------------------------------------------------------------------

@dataclass
class Orientation:
    """Euler angles in radians."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def to_dict(self) -> dict:
        return {
            "yaw": round(self.yaw, 6),
            "pitch": round(self.pitch, 6),
            "roll": round(self.roll, 6),
        }


# ---------------------------------------------------------------------------
# Finger features
# ---------------------------------------------------------------------------

@dataclass
class FingerFeatures:
    """Articulation features for a single finger."""
    joint_angles: list[float] = field(default_factory=list)  # radians at each joint
    curl: float = 0.0           # 0 = fully extended, 1 = fully curled
    spread: float = 0.0         # angle from adjacent finger (radians)
    extension: float = 1.0      # 1 - curl

    def to_dict(self) -> dict:
        return {
            "joint_angles": [round(a, 6) for a in self.joint_angles],
            "curl": round(self.curl, 6),
            "spread": round(self.spread, 6),
            "extension": round(self.extension, 6),
        }


# ---------------------------------------------------------------------------
# Hand data (per frame, per hand)
# ---------------------------------------------------------------------------

@dataclass
class FrameHandData:
    """All hand data for one hand in one frame."""
    detected: bool = False
    landmarks: list[LandmarkPoint] = field(default_factory=list)
    world_landmarks: list[WorldLandmarkPoint] = field(default_factory=list)
    smoothed_landmarks: Optional[list[LandmarkPoint]] = None
    local_landmarks: list[LandmarkPoint] = field(default_factory=list)

    # Derived features
    palm_center: list[float] = field(default_factory=list)
    orientation: Orientation = field(default_factory=Orientation)
    finger_features: dict[str, FingerFeatures] = field(default_factory=dict)
    openness: float = 0.0
    hand_size: float = 0.0
    velocity: list[float] = field(default_factory=list)
    acceleration: list[float] = field(default_factory=list)

    def to_dict(self, include_raw: bool = True) -> dict:
        d: dict = {"detected": self.detected}
        if self.detected:
            d["landmarks"] = [lm.to_dict() for lm in self.landmarks]
            if self.smoothed_landmarks is not None:
                d["smoothed_landmarks"] = [lm.to_dict() for lm in self.smoothed_landmarks]
            if include_raw and self.world_landmarks:
                d["world_landmarks"] = [lm.to_dict() for lm in self.world_landmarks]
            if self.local_landmarks:
                d["local_landmarks"] = [lm.to_dict() for lm in self.local_landmarks]
            d["derived"] = {
                "palm_center": [round(v, 6) for v in self.palm_center],
                "orientation": self.orientation.to_dict(),
                "finger_curl": {k: round(v.curl, 6) for k, v in self.finger_features.items()},
                "finger_spread": {k: round(v.spread, 6) for k, v in self.finger_features.items()},
                "joint_angles": {k: v.to_dict()["joint_angles"] for k, v in self.finger_features.items()},
                "openness": round(self.openness, 6),
                "hand_size": round(self.hand_size, 6),
                "velocity": [round(v, 6) for v in self.velocity],
                "acceleration": [round(v, 6) for v in self.acceleration],
            }
        return d


# ---------------------------------------------------------------------------
# Body data (per frame)
# ---------------------------------------------------------------------------

@dataclass
class FrameBodyData:
    """All body pose data for one frame."""
    detected: bool = False
    landmarks: list[LandmarkPoint] = field(default_factory=list)
    world_landmarks: list[WorldLandmarkPoint] = field(default_factory=list)
    smoothed_landmarks: Optional[list[LandmarkPoint]] = None

    # Derived
    body_center: list[float] = field(default_factory=list)
    shoulder_width: float = 0.0

    def to_dict(self, include_raw: bool = True) -> dict:
        d: dict = {"detected": self.detected}
        if self.detected:
            d["landmarks"] = [lm.to_dict() for lm in self.landmarks]
            if self.smoothed_landmarks is not None:
                d["smoothed_landmarks"] = [lm.to_dict() for lm in self.smoothed_landmarks]
            if include_raw and self.world_landmarks:
                d["world_landmarks"] = [lm.to_dict() for lm in self.world_landmarks]
            d["derived"] = {
                "body_center": [round(v, 6) for v in self.body_center],
                "shoulder_width": round(self.shoulder_width, 6),
            }
        return d


# ---------------------------------------------------------------------------
# Head data (per frame)
# ---------------------------------------------------------------------------

@dataclass
class FrameHeadData:
    """Head position and rotation for one frame."""
    detected: bool = False
    position: list[float] = field(default_factory=list)  # normalized [x, y]
    rotation: Orientation = field(default_factory=Orientation)

    def to_dict(self) -> dict:
        d: dict = {"detected": self.detected}
        if self.detected:
            d["position"] = [round(v, 6) for v in self.position]
            d["rotation"] = self.rotation.to_dict()
        return d


# ---------------------------------------------------------------------------
# Face data (per frame)
# ---------------------------------------------------------------------------

@dataclass
class FrameFaceData:
    """Face landmarks, blendshapes, and derived expression features."""
    detected: bool = False
    landmarks: Optional[list[LandmarkPoint]] = None  # 478 face landmarks (optional)
    blendshapes: dict[str, float] = field(default_factory=dict)
    derived: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_face_landmarks: bool = False) -> dict:
        d: dict = {"detected": self.detected}
        if self.detected:
            if include_face_landmarks and self.landmarks:
                d["landmarks"] = [lm.to_dict() for lm in self.landmarks]
            d["blendshapes"] = {k: round(v, 6) for k, v in self.blendshapes.items()}
            d["derived"] = {k: round(v, 6) for k, v in self.derived.items()}
        return d


# ---------------------------------------------------------------------------
# Full frame data
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """Complete extraction data for a single video frame."""
    frame_index: int
    timestamp: float

    body: FrameBodyData = field(default_factory=FrameBodyData)
    left_hand: FrameHandData = field(default_factory=FrameHandData)
    right_hand: FrameHandData = field(default_factory=FrameHandData)
    head: FrameHeadData = field(default_factory=FrameHeadData)
    face: FrameFaceData = field(default_factory=FrameFaceData)

    # Temporal event (optional)
    motion_phase: Optional[str] = None

    def to_dict(
        self,
        include_raw: bool = True,
        include_face_landmarks: bool = False,
    ) -> dict:
        d: dict = {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 6),
            "body": self.body.to_dict(include_raw=include_raw),
            "left_hand": self.left_hand.to_dict(include_raw=include_raw),
            "right_hand": self.right_hand.to_dict(include_raw=include_raw),
            "head": self.head.to_dict(),
            "face": self.face.to_dict(include_face_landmarks=include_face_landmarks),
        }
        if self.motion_phase is not None:
            d["motion_phase"] = self.motion_phase
        return d


# ---------------------------------------------------------------------------
# Top-level motion data container
# ---------------------------------------------------------------------------

@dataclass
class MotionData:
    """Complete motion extraction result for one video."""
    metadata: VideoMetadata
    frames: list[FrameData] = field(default_factory=list)

    def to_dict(
        self,
        include_raw: bool = True,
        include_face_landmarks: bool = False,
    ) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "frames": [
                f.to_dict(
                    include_raw=include_raw,
                    include_face_landmarks=include_face_landmarks,
                )
                for f in self.frames
            ],
        }
