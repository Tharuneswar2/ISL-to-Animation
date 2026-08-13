"""
Normalization and relative coordinate module.

Computes local/relative coordinates for landmarks in addition to
global normalized coordinates.  This makes the motion representation
more robust to camera resolution and signer position.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from extraction.data_types import FrameData, LandmarkPoint


def compute_body_derived(frame: FrameData) -> None:
    """
    Compute body-level derived features:
    - body center (midpoint of hips)
    - shoulder width
    """
    if not frame.body.detected or len(frame.body.landmarks) < 25:
        return

    lms = frame.body.landmarks

    # Body center: midpoint of left_hip (23) and right_hip (24)
    left_hip = np.array([lms[23].x, lms[23].y])
    right_hip = np.array([lms[24].x, lms[24].y])
    body_center = ((left_hip + right_hip) / 2.0).tolist()
    frame.body.body_center = body_center

    # Shoulder width
    left_shoulder = np.array([lms[11].x, lms[11].y])
    right_shoulder = np.array([lms[12].x, lms[12].y])
    frame.body.shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))


def compute_hand_relative_to_body(frame: FrameData) -> None:
    """
    Compute hand position relative to body landmarks for
    position-invariant representation.

    Adds hand position relative to the shoulder center.
    """
    if not frame.body.detected or len(frame.body.landmarks) < 17:
        return

    lms = frame.body.landmarks

    # Shoulder center
    left_shoulder = np.array([lms[11].x, lms[11].y])
    right_shoulder = np.array([lms[12].x, lms[12].y])
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    for side, hand in [("left", frame.left_hand), ("right", frame.right_hand)]:
        if hand.detected and hand.palm_center:
            palm = np.array(hand.palm_center[:2])
            # Relative position to shoulder center (body-relative)
            rel = (palm - shoulder_center).tolist()
            # Store as extra field in local_landmarks metadata
            # We'll use a convention: add a "body_relative" landmark at the end
            hand.local_landmarks.append(LandmarkPoint(
                x=rel[0], y=rel[1], z=0.0,
                visibility=1.0,
                name=f"{side}_hand_body_relative",
            ))


def compute_arm_vectors(frame: FrameData) -> None:
    """
    Compute arm segment direction vectors for avatar arm orientation.

    Stores upper arm (shoulder→elbow) and forearm (elbow→wrist) directions.
    """
    if not frame.body.detected or len(frame.body.landmarks) < 17:
        return

    lms = frame.body.landmarks

    for side, shoulder_idx, elbow_idx, wrist_idx in [
        ("left", 11, 13, 15),
        ("right", 12, 14, 16),
    ]:
        shoulder = np.array([lms[shoulder_idx].x, lms[shoulder_idx].y])
        elbow = np.array([lms[elbow_idx].x, lms[elbow_idx].y])
        wrist = np.array([lms[wrist_idx].x, lms[wrist_idx].y])

        # Upper arm direction (normalised)
        upper_arm = elbow - shoulder
        upper_norm = np.linalg.norm(upper_arm)
        if upper_norm > 1e-8:
            upper_arm = upper_arm / upper_norm

        # Forearm direction (normalised)
        forearm = wrist - elbow
        forearm_norm = np.linalg.norm(forearm)
        if forearm_norm > 1e-8:
            forearm = forearm / forearm_norm

        # Elbow angle
        if upper_norm > 1e-8 and forearm_norm > 1e-8:
            cos_angle = np.dot(-upper_arm, forearm)
            elbow_angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        else:
            elbow_angle = 0.0

        hand = frame.left_hand if side == "left" else frame.right_hand
        if hand.detected:
            hand.local_landmarks.append(LandmarkPoint(
                x=float(upper_arm[0]), y=float(upper_arm[1]), z=0.0,
                visibility=1.0,
                name=f"{side}_upper_arm_direction",
            ))
            hand.local_landmarks.append(LandmarkPoint(
                x=float(forearm[0]), y=float(forearm[1]), z=0.0,
                visibility=1.0,
                name=f"{side}_forearm_direction",
            ))
            hand.local_landmarks.append(LandmarkPoint(
                x=elbow_angle, y=0.0, z=0.0,
                visibility=1.0,
                name=f"{side}_elbow_angle",
            ))


def apply_normalizations(frames: list[FrameData]) -> None:
    """Apply all normalization passes to a list of frames."""
    for frame in frames:
        compute_body_derived(frame)
        compute_hand_relative_to_body(frame)
        compute_arm_vectors(frame)
