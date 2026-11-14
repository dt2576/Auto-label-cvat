"""CVAT row-major RLE with an inclusive bounding box."""
import numpy as np


def encode_mask(bitmap):
    bitmap = np.asarray(bitmap, dtype=bool)
    if bitmap.ndim != 2:
        raise ValueError('Mask must be two-dimensional')
    ys, xs = np.nonzero(bitmap)
    if not len(xs):
        return None
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    flat = bitmap[top:bottom + 1, left:right + 1].reshape(-1)
    boundaries = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]))
    runs = np.diff(boundaries).tolist()
    if flat[0]:
        runs.insert(0, 0)
    return runs + [left, top, right, bottom]


def validate_mask(points):
    if len(points) < 6 or any(not isinstance(v, (int, float)) or not np.isfinite(v)
                              or v < 0 or int(v) != v for v in points):
        raise ValueError('Invalid CVAT mask RLE')
    *runs, left, top, right, bottom = points
    if right < left or bottom < top or sum(runs) != (right - left + 1) * (bottom - top + 1):
        raise ValueError('CVAT mask size does not match RLE')
