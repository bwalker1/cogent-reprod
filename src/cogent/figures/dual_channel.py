"""Percentile normalization and simple RGB colors for paired module maps."""

import numpy as np

COLOR1 = (1.0, 0.35, 0.05)
COLOR2 = (0.05, 0.35, 1.0)


def normalize_channels(first, second):
    channels = []
    for values in (first, second):
        values = np.asarray(values, dtype=float)
        ordered = np.sort(values)
        lo, hi = ordered[
            np.minimum(
                (len(values) * np.array([0.75, 0.99])).astype(int), len(values) - 1
            )
        ]
        channels.append(
            np.clip((values - lo) / (hi - lo), 0, 1)
            if hi - lo >= 1e-12
            else np.zeros_like(values)
        )
    peaks = []
    for values in channels:
        positive = np.sort(values[values > 0])
        peaks.append(
            positive[min(int(len(positive) * 0.95), len(positive) - 1)]
            if len(positive)
            else 1.0
        )
    if min(peaks) >= 1e-12:
        channels = [
            np.clip(values * (np.mean(peaks) / peak), 0, 1)
            for values, peak in zip(channels, peaks)
        ]
    return channels


def mix_colors(first, second, color1=COLOR1, color2=COLOR2):
    """Blend two scaled signals in RGB; overlap combines their colors."""
    first, second = np.broadcast_arrays(first, second)
    return np.clip(
        first[..., None] * np.asarray(color1) + second[..., None] * np.asarray(color2),
        0,
        1,
    )
