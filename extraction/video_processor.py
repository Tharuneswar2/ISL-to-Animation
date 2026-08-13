"""
Video preprocessing module.

Reads a video file with OpenCV, extracts metadata (FPS, resolution,
frame count, duration), and yields frames in chronological order
with accurate timestamps.
"""

from __future__ import annotations

import os
from typing import Generator

import cv2
import numpy as np

from extraction.data_types import VideoMetadata


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from a video file without reading all frames."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()

    return VideoMetadata(
        source_video=os.path.basename(video_path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration=duration,
    )


def iterate_frames(
    video_path: str,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """
    Yield (frame_index, timestamp_seconds, bgr_frame) for every frame.

    Timestamps are computed from FPS to guarantee monotonic progression.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / fps
        yield frame_index, timestamp, frame
        frame_index += 1

    cap.release()
