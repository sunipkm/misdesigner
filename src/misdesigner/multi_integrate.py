import os, os.path
import ctypes
from ctypes import *
import ctypes.util
from typing import Tuple
import numpy as np
from glob import glob

DIR = os.path.dirname(os.path.abspath(__file__))
LIBROOT = 'integrate_unsort_lib'
LIB = glob(os.path.join(DIR, LIBROOT + '.*'))
if len(LIB) == 0:
    raise FileNotFoundError(f'No shared library found in {DIR} with name {LIBROOT}.*')
LIB = LIB[0]
DLL = ctypes.CDLL(LIB)
if DLL is None:
    raise FileNotFoundError(f'Could not load shared library {LIB}')

DLL.create_data.restype = POINTER(c_void_p)
DLL.create_wavelength_to_rgb.restype = POINTER(c_void_p)
DLL.create_args.restype = POINTER(c_void_p)
DLL.create_unsort.restype = POINTER(c_void_p)


def multi_integrate(x: np.ndarray, y: np.ndarray, low_lim: np.ndarray, high_lim: np.ndarray) -> np.ndarray:
    """## Integrate multiple regions of a 1-D curve.

    ### Args:
        - `x (np.ndarray)`: X values of the curve.
        - `y (np.ndarray)`: Y values of the curve.
        - `low_lim (np.ndarray)`: Lower limits of the regions to integrate. Must be sorted in ascending order.
        - `high_lim (np.ndarray)`: Upper limits of the regions to integrate. Must be sorted in ascending order.

    ### Raises:
        - `ValueError`: X, Y, low_lim, and high_lim must be of type np.float64.
        - `ValueError`: X, Y, low_lim, and high_lim must be 1D.
        - `ValueError`: X and Y must have the same length.
        - `ValueError`: low_lim and high_lim must have the same length.

    ### Returns:
        - `np.ndarray`: The integrated values.
    """
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    if y.dtype != np.float64:
        y = y.astype(np.float64)
    if low_lim.dtype != np.float64:
        low_lim = low_lim.astype(np.float64)
    if high_lim.dtype != np.float64:
        high_lim = high_lim.astype(np.float64)
    if x.ndim != 1:
        raise ValueError('x must be 1D.')
    if y.ndim != 1:
        raise ValueError('y must be 1D.')
    if low_lim.ndim != 1:
        raise ValueError('low_lim must be 1D.')
    if high_lim.ndim != 1:
        raise ValueError('high_lim must be 1D.')
    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')
    if len(low_lim) != len(high_lim):
        raise ValueError('low_lim and high_lim must have the same length.')
    out = np.zeros_like(low_lim, dtype=np.float64)
    data_t = DLL.create_data(x.ctypes.data_as(POINTER(c_double)), y.ctypes.data_as(POINTER(c_double)), c_size_t(len(x)))
    args_t = DLL.create_args(low_lim.ctypes.data_as(POINTER(c_double)), high_lim.ctypes.data_as(POINTER(c_double)), out.ctypes.data_as(POINTER(c_double)), c_size_t(len(low_lim)))
    DLL.integrate(data_t, args_t)
    DLL.free_data(data_t)
    DLL.free_args(args_t)
    return out

def unsort(x: np.ndarray, args: np.ndarray)->np.ndarray:
    """## Unsort a 1-D array.

    ### Args:
        - `x (np.ndarray)`: The sorted array.
        - `args (np.ndarray)`: The argsort of the original, unsorted array.

    ### Raises:
        - `ValueError`: args must be of type int.
        - `ValueError`: X and args must be 1D.
        - `ValueError`: X and args must have the same length.

    ### Returns:
        - `np.ndarray`: The output, unsorted array.
    """
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    if args.dtype != int:
        raise ValueError('args must be of type int.')
    if x.ndim != 1:
        raise ValueError('x must be 1D.')
    if args.ndim != 1:
        raise ValueError('args must be 1D.')
    if len(x) != len(args):
        raise ValueError('x and args must have the same length.')
    out = np.zeros_like(x, dtype=np.float64)
    args = args.astype(c_size_t)
    unsort_t = DLL.create_unsort(x.ctypes.data_as(POINTER(c_double)), out.ctypes.data_as(POINTER(c_double)), args.ctypes.data_as(POINTER(c_size_t)), c_size_t(len(x)))
    DLL.unsort(unsort_t)
    DLL.free_unsort(unsort_t)
    return out

def wavelength_to_rgb(lam: np.ndarray, gamma: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """## Convert a wavelength to an RGB color.

    ### Args:
        - `lam (np.ndarray)`: The wavelength to convert.
        - `gamma (float)`: The gamma correction factor.

    ### Raises:
        - `ValueError`: lam must be of type np.float64.
        - `ValueError`: lam must be 1D.

    ### Returns:
        - `np.ndarray`: The RGB color.
    """
    if lam.dtype != np.float64:
        lam = lam.astype(np.float64)
    shape = lam.shape
    lam = lam.flatten()
    if gamma < 0:
        gamma = 0
    if gamma > 1:
        gamma = 1
    r = np.zeros_like(lam, dtype=np.float64)
    g = np.zeros_like(lam, dtype=np.float64)
    b = np.zeros_like(lam, dtype=np.float64)
    data_t = DLL.create_wavelength_to_rgb(lam.ctypes.data_as(POINTER(c_double)),
                                r.ctypes.data_as(POINTER(c_double)),
                                g.ctypes.data_as(POINTER(c_double)),
                                b.ctypes.data_as(POINTER(c_double)),
                                c_double(gamma),
                                c_size_t(len(lam)),
                             )
    DLL.wavelength_to_rgb(data_t)
    DLL.free_wavelength_to_rgb(data_t)
    r = r.reshape(shape)
    g = g.reshape(shape)
    b = b.reshape(shape)
    return r, g, b
