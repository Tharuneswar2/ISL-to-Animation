"""
Debug visualization module.

Generates diagnostic overlay videos and analysis plots to verify
extraction quality before using the motion data downstream.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from extraction.data_types import FrameData, VideoMetadata

# MediaPipe pose skeleton connections (index pairs)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # left eye → ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # right eye → ear
    (9, 10),                           # mouth
    (11, 12),                          # shoulders
    (11, 13), (13, 15),               # left arm
    (12, 14), (14, 16),               # right arm
    (11, 23), (12, 24),               # torso
    (23, 24),                          # hips
    (23, 25), (25, 27),               # left leg
    (24, 26), (26, 28),               # right leg
    (15, 17), (15, 19), (15, 21),     # left hand connections
    (16, 18), (16, 20), (16, 22),     # right hand connections
    (27, 29), (27, 31),               # left foot
    (28, 30), (28, 32),               # right foot
]

# Hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),            # palm
]

# Colors
COLOR_POSE = (0, 255, 0)        # green
COLOR_LEFT_HAND = (255, 128, 0) # orange
COLOR_RIGHT_HAND = (0, 128, 255)# blue
COLOR_FACE = (200, 200, 200)    # light gray
COLOR_TEXT = (255, 255, 255)     # white
COLOR_BG = (0, 0, 0)            # black


def generate_debug_outputs(
    video_path: str,
    frames: list[FrameData],
    metadata: VideoMetadata,
    output_dir: str,
) -> None:
    """
    Generate all debug outputs: overlay videos and analysis plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[debug] Generating debug outputs in {output_dir}")

    # Generate overlay videos
    _generate_skeleton_video(video_path, frames, metadata, output_dir)
    _generate_hand_video(video_path, frames, metadata, output_dir)
    _generate_face_video(video_path, frames, metadata, output_dir)

    # Generate plots
    _plot_wrist_trajectory(frames, metadata, output_dir)
    _plot_hand_velocity(frames, metadata, output_dir)
    _plot_finger_curl(frames, metadata, output_dir)
    _plot_head_rotation(frames, metadata, output_dir)

    print(f"[debug] Debug outputs complete.")


def _generate_skeleton_video(
    video_path: str,
    frames: list[FrameData],
    metadata: VideoMetadata,
    output_dir: str,
) -> None:
    """Generate skeleton overlay video."""
    output_path = os.path.join(output_dir, "skeleton_preview.mp4")
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )

    frame_idx = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        if frame_idx < len(frames):
            fd = frames[frame_idx]

            # Draw pose skeleton
            if fd.body.detected:
                lms = fd.body.landmarks
                w, h = metadata.width, metadata.height

                # Draw connections
                for i, j in POSE_CONNECTIONS:
                    if i < len(lms) and j < len(lms):
                        if lms[i].visibility > 0.3 and lms[j].visibility > 0.3:
                            p1 = (int(lms[i].x * w), int(lms[i].y * h))
                            p2 = (int(lms[j].x * w), int(lms[j].y * h))
                            cv2.line(img, p1, p2, COLOR_POSE, 2)

                # Draw landmarks
                for lm in lms:
                    if lm.visibility > 0.3:
                        pt = (int(lm.x * w), int(lm.y * h))
                        cv2.circle(img, pt, 3, COLOR_POSE, -1)

            # Draw info text
            _draw_frame_info(img, fd)

        writer.write(img)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[debug] Wrote {output_path}")


def _generate_hand_video(
    video_path: str,
    frames: list[FrameData],
    metadata: VideoMetadata,
    output_dir: str,
) -> None:
    """Generate hand landmarks overlay video."""
    output_path = os.path.join(output_dir, "hand_preview.mp4")
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )

    frame_idx = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        if frame_idx < len(frames):
            fd = frames[frame_idx]
            w, h = metadata.width, metadata.height

            # Draw left hand
            if fd.left_hand.detected:
                _draw_hand(img, fd.left_hand.landmarks, w, h, COLOR_LEFT_HAND, "L")

            # Draw right hand
            if fd.right_hand.detected:
                _draw_hand(img, fd.right_hand.landmarks, w, h, COLOR_RIGHT_HAND, "R")

            # Info
            info_lines = [
                f"Frame: {fd.frame_index}  t={fd.timestamp:.3f}s",
                f"L: {'Y' if fd.left_hand.detected else 'N'}  "
                f"R: {'Y' if fd.right_hand.detected else 'N'}",
            ]
            if fd.left_hand.detected:
                info_lines.append(f"L open={fd.left_hand.openness:.2f}")
            if fd.right_hand.detected:
                info_lines.append(f"R open={fd.right_hand.openness:.2f}")

            for i, line in enumerate(info_lines):
                cv2.putText(
                    img, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1,
                )

        writer.write(img)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[debug] Wrote {output_path}")


def _draw_hand(img, landmarks, w, h, color, label):
    """Draw hand landmarks and connections."""
    if len(landmarks) < 21:
        return

    # Draw connections
    for i, j in HAND_CONNECTIONS:
        if i < len(landmarks) and j < len(landmarks):
            p1 = (int(landmarks[i].x * w), int(landmarks[i].y * h))
            p2 = (int(landmarks[j].x * w), int(landmarks[j].y * h))
            cv2.line(img, p1, p2, color, 2)

    # Draw points
    for lm in landmarks:
        pt = (int(lm.x * w), int(lm.y * h))
        cv2.circle(img, pt, 3, color, -1)

    # Label
    wrist = landmarks[0]
    cv2.putText(
        img, label,
        (int(wrist.x * w) - 10, int(wrist.y * h) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )


def _generate_face_video(
    video_path: str,
    frames: list[FrameData],
    metadata: VideoMetadata,
    output_dir: str,
) -> None:
    """Generate face mesh overlay video."""
    output_path = os.path.join(output_dir, "face_preview.mp4")
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )

    frame_idx = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        if frame_idx < len(frames):
            fd = frames[frame_idx]
            w, h = metadata.width, metadata.height

            # Draw face landmarks if available
            if fd.face.detected and fd.face.landmarks:
                for lm in fd.face.landmarks:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(img, pt, 1, COLOR_FACE, -1)

            # Draw face feature indicators
            if fd.face.detected and fd.face.derived:
                info = fd.face.derived
                info_lines = [
                    f"Frame: {fd.frame_index}  t={fd.timestamp:.3f}s",
                    f"Eye L: {info.get('eye_left_open', 0):.2f}  "
                    f"R: {info.get('eye_right_open', 0):.2f}",
                    f"Mouth: {info.get('mouth_open', 0):.2f}  "
                    f"Smile: {info.get('smile', 0):.2f}",
                    f"Brow L: {info.get('eyebrow_left', 0):.2f}  "
                    f"R: {info.get('eyebrow_right', 0):.2f}",
                ]
                for i, line in enumerate(info_lines):
                    cv2.putText(
                        img, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1,
                    )

            # Head rotation
            if fd.head.detected:
                rot = fd.head.rotation
                rot_text = (
                    f"Head P={rot.pitch:.2f} Y={rot.yaw:.2f} R={rot.roll:.2f}"
                )
                cv2.putText(
                    img, rot_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )

        writer.write(img)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[debug] Wrote {output_path}")


def _draw_frame_info(img: np.ndarray, fd: FrameData) -> None:
    """Draw frame number, timestamp, and phase on the image."""
    h, w = img.shape[:2]
    lines = [
        f"Frame: {fd.frame_index}  t={fd.timestamp:.3f}s",
    ]
    if fd.motion_phase:
        lines.append(f"Phase: {fd.motion_phase}")

    for i, line in enumerate(lines):
        cv2.putText(
            img, line, (10, 25 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1,
        )


# ---------------------------------------------------------------------------
# Analysis plots
# ---------------------------------------------------------------------------

def _plot_wrist_trajectory(
    frames: list[FrameData],
    metadata: VideoMetadata,
    output_dir: str,
) -> None:
    """Plot wrist x/y trajectories over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    times = [f.timestamp for f in frames]

    for side, color, label in [("left", "tab:orange", "Left"), ("right", "tab:blue", "Right")]:
        xs, ys, ts = [], [], []
        for f in frames:
            hand = f.left_hand if side == "left" else f.right_hand
            if hand.detected and hand.palm_center:
                ts.append(f.timestamp)
                xs.append(hand.palm_center[0])
                ys.append(hand.palm_center[1])

        if ts:
            axes[0].plot(ts, xs, color=color, label=f"{label} Wrist X", alpha=0.8)
            axes[1].plot(ts, ys, color=color, label=f"{label} Wrist Y", alpha=0.8)

    axes[0].set_ylabel("Normalized X")
    axes[0].legend()
    axes[0].set_title("Wrist Trajectory")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Normalized Y")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "wrist_trajectory.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[debug] Wrote {path}")


def _plot_hand_velocity(
    frames: list[FrameData],
    metadata: VideoMetadata,
    output_dir: str,
) -> None:
    """Plot hand velocity magnitude over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for side, color, label in [("left", "tab:orange", "Left"), ("right", "tab:blue", "Right")]:
        ts, speeds = [], []
        for f in frames:
            hand = f.left_hand if side == "left" else f.right_hand
            if hand.detected and hand.velocity:
                ts.append(f.timestamp)
                speeds.append(float(np.linalg.norm(hand.velocity)))

        if ts:
            ax.plot(ts, speeds, color=color, label=f"{label} Hand", alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (norm units/s)")
    ax.set_title("Hand Velocity Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "hand_velocity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[debug] Wrote {path}")


def _plot_finger_curl(
    frames: list[FrameData],
    metadata: VideoMetadata,
    output_dir: str,
) -> None:
    """Plot finger curl values over time for both hands."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]

    for ax_idx, (side, title) in enumerate([("left", "Left Hand"), ("right", "Right Hand")]):
        ax = axes[ax_idx]
        for finger, color in zip(fingers, colors):
            ts, curls = [], []
            for f in frames:
                hand = f.left_hand if side == "left" else f.right_hand
                if hand.detected and finger in hand.finger_features:
                    ts.append(f.timestamp)
                    curls.append(hand.finger_features[finger].curl)
            if ts:
                ax.plot(ts, curls, color=color, label=finger.capitalize(), alpha=0.8)

        ax.set_ylabel("Curl (0=open, 1=closed)")
        ax.set_title(f"{title} Finger Curl")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    path = os.path.join(output_dir, "finger_curl.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[debug] Wrote {path}")


def _plot_head_rotation(
    frames: list[FrameData],
    metadata: VideoMetadata,
    output_dir: str,
) -> None:
    """Plot head rotation (pitch/yaw/roll) over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ts = [f.timestamp for f in frames if f.head.detected]
    pitches = [f.head.rotation.pitch for f in frames if f.head.detected]
    yaws = [f.head.rotation.yaw for f in frames if f.head.detected]
    rolls = [f.head.rotation.roll for f in frames if f.head.detected]

    if ts:
        ax.plot(ts, pitches, label="Pitch", color="tab:red", alpha=0.8)
        ax.plot(ts, yaws, label="Yaw", color="tab:green", alpha=0.8)
        ax.plot(ts, rolls, label="Roll", color="tab:blue", alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rotation (radians)")
    ax.set_title("Head Rotation Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "head_rotation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[debug] Wrote {path}")
