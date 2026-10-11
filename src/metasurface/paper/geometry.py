"""Explicit approximations to the eight 8-by-8 tile drawings in Figure 1."""

import numpy as np


def masks():
    """Return six square and two diamond ring masks in stable ID order.

    Returns:
        ndarray: Binary masks with shape (8, 8, 8).
    """
    tiles = []
    for side, width in ((4, 1), (6, 1), (6, 2), (8, 1), (8, 2), (8, 3)):
        tile = np.zeros((8, 8), dtype=np.uint8)
        start = (8 - side) // 2
        tile[start:start + side, start:start + side] = 1
        tile[start + width:start + side - width, start + width:start + side - width] = 0
        tiles.append(tile)
    y, x = np.indices((8, 8))
    distance = abs(x - 3.5) + abs(y - 3.5)
    tiles.extend(((distance >= low) & (distance <= high)).astype(np.uint8)
                 for low, high in ((3, 4), (5, 6)))
    return np.stack(tiles)


def sample_codes(count=2000, seed=2026):
    """Draw distinct 16-position tile assignments reproducibly.

    Args:
        count (int): Number of unique layouts.
        seed (int): NumPy random seed.

    Returns:
        ndarray: Tile IDs with shape (count, 16).

    Raises:
        ValueError: If count is not positive.
    """
    if not isinstance(count, int) or count < 1:
        raise ValueError("Count must be a positive integer.")
    rng = np.random.default_rng(seed)
    rows = []
    seen = set()
    while len(rows) < count:
        row = tuple(int(v) for v in rng.integers(0, 8, size=16))
        if row not in seen:
            seen.add(row)
            rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def pixel_layout(codes):
    """Build an unrestricted 4-by-4 tile arrangement from 16 IDs.

    Args:
        codes (array-like): Sixteen integers from zero through seven.

    Returns:
        ndarray: Binary 32-by-32 copper mask.

    Raises:
        ValueError: If tile IDs are invalid.
    """
    values = np.asarray(codes)
    if (values.shape != (16,) or not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all()
            or np.any((values < 0) | (values > 7))):
        raise ValueError("Expected sixteen tile IDs from zero through seven.")
    tiles = masks()
    return np.block([[tiles[int(values[4 * row + col])] for col in range(4)]
                     for row in range(4)])


def rectangles(codes, pixel_m):
    """Merge copper pixels into nonoverlapping horizontal SI rectangles.

    Args:
        codes (array-like): Sixteen tile IDs.
        pixel_m (float): Pixel side length in metres.

    Returns:
        list[tuple]: Rectangle center and spans (x, y, dx, dy).

    Raises:
        ValueError: If pixel size or codes are invalid.
    """
    if not np.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError("Pixel size must be positive.")
    mask = pixel_layout(codes)
    result = []
    for y, row in enumerate(mask):
        changes = np.diff(np.r_[0, row.astype(int), 0])
        for left, right in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            result.append(((left + right) * pixel_m / 2 - 16 * pixel_m,
                           (y + 0.5 - 16) * pixel_m, (right - left) * pixel_m, pixel_m))
    return result
