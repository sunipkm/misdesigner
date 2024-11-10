from __future__ import annotations
from typing import Iterable, List, SupportsFloat as Numeric, Tuple
import numpy as np
import astropy.io.fits as fits


def open_fits(fn: str, timestamp: bool = True) -> np.ndarray:
    """opens fits file using astropy.io.fits
    Args:
        fn (str): file path.

    Returns:
        tuple (np.array,np.array): image of shape (n,m) , list of headers.
    """
    with fits.open(fn) as hdul:
        idx = 0
        if 'COMPRESSED_IMAGE' in list(hdul[idx].header.keys()):
            idx = 1
        data = hdul[idx].data
    if timestamp:
        h_keys = list(hdul[idx].header.keys())
        if 'TIMESTAMP_S' in h_keys:
            tstamp = hdul[idx].header['TIMESTAMP_S']
            tstamp += hdul[idx].header['TIMESTAMP_NS'] * 1e-9
        elif 'TIMESTAMP' in h_keys:
            tstamp = hdul[idx].header['TIMESTAMP']
        elif 'TSTAMP' in h_keys:
            tstamp = hdul[idx].header['TSTAMP']
        else:
            tstamp = None
        return data, tstamp
    else:
        return data


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