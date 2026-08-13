"""
Temporal smoothing using the One Euro Filter.

The One Euro Filter adaptively adjusts its cutoff frequency based on
signal speed: slow movements get heavy smoothing (jitter removal),
fast movements get light smoothing (lag reduction).  This is ideal
for ISL hand tracking where fast articulation must be preserved.

Reference:
    Casiez, Roussel, Vogel — 1€ Filter (CHI 2012)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from extraction.data_types import LandmarkPoint


class OneEuroFilter:
    """One Euro low-pass filter for a single scalar signal."""

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 0:
            return x  # degenerate case — no time elapsed

        # Filtered derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filtered value
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class LandmarkSmoother:
    """
    Applies One Euro filtering independently to each coordinate
    of each landmark across time.

    Supports per-body-part tuning via preset profiles.
    """

    # Smoothing presets — (min_cutoff, beta)
    PRESETS = {
        "body":    (1.5, 0.004),   # heavier smoothing for stable body
        "hand":    (1.0, 0.01),    # lighter smoothing, preserve fast fingers
        "head":    (1.2, 0.005),   # moderate smoothing for head
        "face":    (1.5, 0.003),   # heavier smoothing for face
    }

    def __init__(self, preset: str = "body") -> None:
        min_cutoff, beta = self.PRESETS.get(preset, self.PRESETS["body"])
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[tuple[int, str], OneEuroFilter] = {}

    def smooth(
        self,
        timestamp: float,
        landmarks: list[LandmarkPoint],
    ) -> list[LandmarkPoint]:
        """
        Apply smoothing to a list of landmarks for a single frame.

        Returns a new list of smoothed LandmarkPoints.
        Initialises filters on first call.
        """
        smoothed: list[LandmarkPoint] = []

        for idx, lm in enumerate(landmarks):
            sx = self._filter_value(idx, "x", timestamp, lm.x)
            sy = self._filter_value(idx, "y", timestamp, lm.y)
            sz = self._filter_value(idx, "z", timestamp, lm.z)

            smoothed.append(LandmarkPoint(
                x=sx, y=sy, z=sz,
                visibility=lm.visibility,
                name=lm.name,
            ))

        return smoothed

    def _filter_value(
        self,
        landmark_idx: int,
        coord: str,
        t: float,
        value: float,
    ) -> float:
        key = (landmark_idx, coord)
        if key not in self._filters:
            self._filters[key] = OneEuroFilter(
                t0=t, x0=value,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
            )
            return value
        return self._filters[key](t, value)

    def reset(self) -> None:
        """Clear all filter state."""
        self._filters.clear()


def interpolate_landmarks_gap(
    frames_landmarks: list[Optional[list[LandmarkPoint]]],
    max_gap: int = 3,
) -> list[Optional[list[LandmarkPoint]]]:
    """
    Fill short gaps (≤ max_gap frames) via linear interpolation.

    Longer gaps remain None.
    """
    n = len(frames_landmarks)
    result = list(frames_landmarks)  # shallow copy

    i = 0
    while i < n:
        if result[i] is None:
            # Find gap boundaries
            gap_start = i
            while i < n and result[i] is None:
                i += 1
            gap_end = i  # first valid frame after gap

            gap_len = gap_end - gap_start

            # Only interpolate short gaps with valid neighbours
            if gap_len <= max_gap and gap_start > 0 and gap_end < n:
                before = result[gap_start - 1]
                after = result[gap_end]
                if before is not None and after is not None:
                    for j in range(gap_len):
                        t = (j + 1) / (gap_len + 1)
                        interpolated = _lerp_landmarks(before, after, t)
                        result[gap_start + j] = interpolated
        else:
            i += 1

    return result


def _lerp_landmarks(
    a: list[LandmarkPoint],
    b: list[LandmarkPoint],
    t: float,
) -> list[LandmarkPoint]:
    """Linearly interpolate between two landmark lists."""
    result: list[LandmarkPoint] = []
    for la, lb in zip(a, b):
        result.append(LandmarkPoint(
            x=la.x + (lb.x - la.x) * t,
            y=la.y + (lb.y - la.y) * t,
            z=la.z + (lb.z - la.z) * t,
            visibility=min(la.visibility, lb.visibility) * 0.5,  # low confidence for interpolated
            name=la.name,
        ))
    return result
