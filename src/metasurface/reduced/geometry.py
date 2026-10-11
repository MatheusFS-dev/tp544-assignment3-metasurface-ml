"""Four square-ring tiles and the approved mirrored unit-cell family."""

from itertools import product

import numpy as np


def all_codes():
    """Enumerate the complete reduced geometry space in a stable order.

    Args:
        None.
    Returns:
        ndarray: Integer array of shape (256, 4), with columns A, B, C, D.
    Raises:
        None.
    """
    return np.asarray(list(product(range(4), repeat=4)), dtype=np.int64)


def pixel_layout(codes):
    """Build the 32-by-32 copper mask from four independent pattern choices.

    Args:
        codes (array-like): Four integer pattern IDs from 0 through 3.
    Returns:
        ndarray: Binary copper mask with shape (32, 32).
    Raises:
        ValueError: If codes are not four valid integers.
    Notes:
        Rows correspond to increasing y. Pattern 0 is empty. Ring outer sides
        are 4, 6, and 6 pixels, with widths 1, 1, and 2 pixels respectively.
    Examples:
        pixel_layout([1, 2, 3, 0]).shape == (32, 32)
    """
    values = np.asarray(codes)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Expected four finite pattern IDs.")
    if not np.equal(values, values.astype(int)).all() or np.any((values < 0) | (values > 3)):
        raise ValueError("Pattern IDs must be integers from 0 to 3.")
    tiles = [np.zeros((8, 8), dtype=np.uint8)]
    for side, width in [(4, 1), (6, 1), (6, 2)]:
        tile = np.zeros((8, 8), dtype=np.uint8)
        start = (8 - side) // 2
        tile[start:start + side, start:start + side] = 1
        tile[start + width:start + side - width, start + width:start + side - width] = 0
        tiles.append(tile)
    a, b, c, d = values.astype(int)
    ids = [[a, b, b, a], [c, d, d, c], [c, d, d, c], [a, b, b, a]]
    return np.block([[tiles[i] for i in row] for row in ids])


def rectangles(codes, pixel_m):
    """Merge horizontal copper runs into non-overlapping SI rectangles.

    Args:
        codes (array-like): Four independent pattern IDs.
        pixel_m (float): Positive pixel side length in metres.
    Returns:
        list[tuple]: Rectangle centers and spans as (x, y, dx, dy), in metres.
    Raises:
        ValueError: If codes or pixel size are invalid.
    Notes:
        Horizontal merging reduces solver object count without changing area.
    """
    if not np.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError("Pixel length must be positive.")
    mask = pixel_layout(codes)
    result = []
    for y, row in enumerate(mask):
        changes = np.diff(np.r_[0, row.astype(int), 0])
        for left, right in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            result.append(((left + right) * pixel_m / 2 - 16 * pixel_m,
                           (y + 0.5 - 16) * pixel_m,
                           (right - left) * pixel_m, pixel_m))
    return result
