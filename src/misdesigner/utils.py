from __future__ import annotations
from typing import Iterable, List, SupportsFloat as Numeric, Tuple
import numpy as np

def find_nearest(array: Iterable, targetval: Numeric) -> tuple[int, Numeric]:
    """finds the index and value in an array nearest to the target value.

    Args:
        array (Iterable): array to search.
        targetval (float): target value.

    Returns:
        tuple[int,float]: idex and value. Returns Iterables for both if there is more than one idx.
    """
    dif = np.abs(np.array(array)-targetval)
    idx = np.nanargmin(dif)
    return idx, array[idx]

def common_range(r1: List[Tuple[int, int]], r2: Tuple[int, int]) -> List[Tuple[int, int]]:
    """finds the common range between two ranges."""
    out = []
    for r in r1:
        rmin = max(r[0], r2[0])
        rmax = min(r[1], r2[1])
        if rmin < rmax:
            out.append((rmin, rmax))
    return out

def sign_floor(x):
    return np.sign(x) * np.floor(np.abs(x))

def sign_ceil(x):
    return np.sign(x) * np.ceil(np.abs(x))