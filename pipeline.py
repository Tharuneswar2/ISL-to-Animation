#!/usr/bin/env python3
"""
ISL Motion Extraction Pipeline
===============================

Extracts body pose, hand landmarks, facial expressions, and head rotation
from Indian Sign Language videos and produces a comprehensive motion-data
JSON file suitable for avatar animation.

Usage:
    python pipeline.py --input input/hello_isl.mp4 --output output/motion/hello_motion.json
    python pipeline.py --input input/hello_isl.mp4 --output output/motion/hello_motion.json --debug
    python pipeline.py --input input/hello_isl.mp4 --output output/motion/hello_motion.json --debug --save-raw --save-face-landmarks

Flags:
    --input                 Input video path (required)
    --output                Output JSON path (required)
    --debug                 Generate debug overlay videos and plots
    --smoothing             Apply One Euro temporal smoothing (default: enabled)
    --no-smoothing          Disable temporal smoothing
    --save-raw              Include raw/world landmarks in JSON
    --save-face-landmarks   Include all 478 face landmarks in JSON
    --confidence-threshold  Minimum landmark visibility threshold (default: 0.3)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from extraction.video_processor import get_video_metadata, iterate_frames
from extraction.pose_extractor import HolisticExtractor
from extraction.hand_extractor import extract_hand_features
from extraction.face_extractor import extract_face_features
from extraction.head_extractor import extract_head_pose
from extraction.tracking import TrackingState, filter_low_confidence_landmarks
from extraction.smoothing import LandmarkSmoother
from extraction.features import compute_frame_velocities, detect_motion_phases
from extraction.normalization import apply_normalizations
from extraction.exporter import export_motion_json, export_summary_json
from extraction.data_types import (
    FrameData,
    FrameBodyData,
    FrameHandData,
    FrameFaceData,
    FrameHeadData,
    MotionData,
    LandmarkPoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISL Motion Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input ISL video file",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for output motion JSON file",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Generate debug overlay videos and analysis plots",
    )
    parser.add_argument(
        "--smoothing", action="store_true", default=True,
        help="Apply One Euro temporal smoothing (default: enabled)",
    )
    parser.add_argument(
        "--no-smoothing", action="store_true",
        help="Disable temporal smoothing",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Include raw/world landmarks in JSON output",
    )
    parser.add_argument(
        "--save-face-landmarks", action="store_true",
        help="Include all 478 face landmarks in JSON output",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.3,
        help="Minimum landmark visibility threshold (default: 0.3)",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full extraction pipeline."""
    start_time = time.time()

    enable_smoothing = args.smoothing and not args.no_smoothing

    # ---- Step 1: Video metadata ----
    print("=" * 60)
    print("ISL MOTION EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Smoothing: {enable_smoothing}")
    print(f"Debug: {args.debug}")
    print()

    metadata = get_video_metadata(args.input)
    print(f"[video] {metadata.source_video}")
    print(f"[video] {metadata.width}x{metadata.height} @ {metadata.fps:.2f} FPS")
    print(f"[video] {metadata.frame_count} frames, {metadata.duration:.2f}s")
    print()

    # ---- Step 2: Initialise extractors ----
    print("[init] Loading MediaPipe HolisticLandmarker...")
    extractor = HolisticExtractor(
        min_pose_confidence=args.confidence_threshold,
        min_hand_confidence=args.confidence_threshold,
        min_face_confidence=args.confidence_threshold,
    )

    # Smoothers for different body parts
    body_smoother = LandmarkSmoother(preset="body") if enable_smoothing else None
    left_hand_smoother = LandmarkSmoother(preset="hand") if enable_smoothing else None
    right_hand_smoother = LandmarkSmoother(preset="hand") if enable_smoothing else None

    tracking = TrackingState()
    frames: list[FrameData] = []

    # ---- Step 3: Process frames ----
    print("[extract] Processing frames...")
    processed = 0

    try:
        for frame_idx, timestamp, bgr_frame in iterate_frames(args.input):
            timestamp_ms = int(timestamp * 1000)

            # Run holistic detection
            result = extractor.process_frame(bgr_frame, timestamp_ms)

            # --- Body pose ---
            pose_landmarks = result.get_pose_landmarks()
            pose_world_landmarks = result.get_pose_world_landmarks()
            pose_detected = len(pose_landmarks) > 0

            body_data = FrameBodyData(
                detected=pose_detected,
                landmarks=pose_landmarks,
                world_landmarks=pose_world_landmarks,
            )

            # Apply smoothing to body
            if body_smoother and pose_detected:
                body_data.smoothed_landmarks = body_smoother.smooth(timestamp, pose_landmarks)

            # --- Left hand ---
            left_hand_lms = result.get_hand_landmarks("left")
            left_hand_world = result.get_hand_world_landmarks("left")
            left_detected = len(left_hand_lms) >= 21

            if left_detected:
                left_hand_data = extract_hand_features(left_hand_lms, left_hand_world)
                if left_hand_smoother:
                    left_hand_data.smoothed_landmarks = left_hand_smoother.smooth(
                        timestamp, left_hand_lms,
                    )
            else:
                left_hand_data = FrameHandData(detected=False)

            # --- Right hand ---
            right_hand_lms = result.get_hand_landmarks("right")
            right_hand_world = result.get_hand_world_landmarks("right")
            right_detected = len(right_hand_lms) >= 21

            if right_detected:
                right_hand_data = extract_hand_features(right_hand_lms, right_hand_world)
                if right_hand_smoother:
                    right_hand_data.smoothed_landmarks = right_hand_smoother.smooth(
                        timestamp, right_hand_lms,
                    )
            else:
                right_hand_data = FrameHandData(detected=False)

            # --- Face ---
            face_landmarks = result.get_face_landmarks()
            face_blendshapes = result.get_face_blendshapes()
            face_detected = len(face_landmarks) > 0

            if face_detected:
                face_data = extract_face_features(
                    face_landmarks,
                    face_blendshapes,
                    include_landmarks=args.save_face_landmarks,
                )
            else:
                face_data = FrameFaceData(detected=False)

            # --- Head ---
            head_data = extract_head_pose(
                face_landmarks,
                pose_landmarks,
                metadata.width,
                metadata.height,
            )

            # --- Assemble frame data ---
            frame_data = FrameData(
                frame_index=frame_idx,
                timestamp=timestamp,
                body=body_data,
                left_hand=left_hand_data,
                right_hand=right_hand_data,
                head=head_data,
                face=face_data,
            )
            frames.append(frame_data)

            # Update tracking
            tracking.update(
                pose=pose_detected,
                left_hand=left_detected,
                right_hand=right_detected,
                face=face_detected,
            )

            processed += 1
            if processed % 50 == 0 or processed == metadata.frame_count:
                pct = processed / max(metadata.frame_count, 1) * 100
                print(f"  [{processed}/{metadata.frame_count}] {pct:.0f}%")

    finally:
        extractor.close()

    print(f"[extract] Processed {processed} frames.")
    print()

    # ---- Step 4: Compute normalizations ----
    print("[normalize] Computing relative coordinates...")
    apply_normalizations(frames)

    # ---- Step 5: Compute motion features ----
    print("[features] Computing velocity and acceleration...")
    compute_frame_velocities(frames, metadata.fps)

    # ---- Step 6: Detect motion phases ----
    print("[features] Detecting motion phases...")
    detect_motion_phases(frames, metadata.fps)

    # ---- Step 7: Export ----
    print("[export] Writing JSON...")
    motion_data = MotionData(metadata=metadata, frames=frames)

    export_motion_json(
        motion_data,
        args.output,
        include_raw=args.save_raw,
        include_face_landmarks=args.save_face_landmarks,
    )

    tracking_summary = tracking.get_summary()
    export_summary_json(motion_data, tracking_summary, args.output)

    # ---- Step 8: Debug outputs ----
    if args.debug:
        print()
        print("[debug] Generating debug outputs...")
        from extraction.debug import generate_debug_outputs

        debug_dir = os.path.join(os.path.dirname(args.output), "..", "debug")
        debug_dir = os.path.normpath(debug_dir)
        generate_debug_outputs(args.input, frames, metadata, debug_dir)

    # ---- Done ----
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"Output: {os.path.abspath(args.output)}")
    print()

    # Print tracking summary
    summary = tracking.get_summary()
    print("Detection rates:")
    for key, val in summary.items():
        if key != "total_frames":
            print(f"  {key}: {val:.1f}%")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    run_pipeline(args)


if __name__ == "__main__":
    main()
