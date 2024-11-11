# %%
from __future__ import annotations
import os
from typing import Dict, List, Optional, SupportsFloat as Numeric, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MisGrating:
    """## Instrument Optics Parameters
    """
    fl_collimator: float  # collimator focal length (mm)
    fl_mosaic: float  # grating camera focal length (mm)
    sigma: float  # groove density (grooves/mm)
    # slit X center, Y center, width, height
    slits: Dict[str, MisSlit]
    # mosaic X center, Y center, width, height
    mosaic: MisMosaic


@dataclass
class MisMosaic:
    """## Instrument Mosaic Parameters
    The mosaic itself is defined in the instrument coordinate system.
    The mosaic window is defined in the mosaic coordinate system.
    The origin of the mosaic coordinate system is at the bottom left corner of the mosaic.
    """
    x: float
    y: float
    width: float
    height: float
    windows: List[MisMosaicFilter]

    def __init__(self, x: float, y: float, width: float, height: float, windows: Optional[List[MisMosaicFilter]] = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        if windows is None:
            windows = []
        self.windows = windows


@dataclass
class MisMosaicFilter:
    """## Instrument Mosaic Filter Parameters
    Defined in the mosaic coordinate system.
    The origin of the mosaic coordinate system is at the bottom left corner of the mosaic.
    i.e., the mosaic filter at x, y with width and height is on the mosaic at: (x, y), (x+width, y), (x+width, y+height), (x, y+height)
    """
    x: float
    y: float
    width: float
    height: float
    ranges: List[List[int]]
    name: Optional[str] = None

    def __init__(self, x: float, y: float, width: float, height: float, ranges: List[Tuple[int, int]] = [], name: Optional[str] = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = name
        frange = []
        for range in ranges:
            if len(range) != 2:
                continue
            if range[0] > range[1]:
                range = [range[1], range[0]]
            else:
                range = [range[0], range[1]]
            frange.append(range)
        self.ranges = frange

    def get_xrange(self):
        return (-self.x, -self.x - self.width)
    
    def get_yrange(self):
        return (self.y, self.y + self.height)

    def check_position(self, wavelength: int, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        valid = np.full(beta.shape, False)
        for range in self.ranges:
            if range[0] <= wavelength <= range[1]:
                valid[:] = True
                break
            # else:
            #     print(f'Wavelength {wavelength} not in range {range}.')
        valid &= (-self.x >= beta) & (beta >= -self.x - self.width)
        valid &= (self.y <= gamma) & (gamma <= self.y + self.height)
        return valid


@dataclass
class MisSlit:
    """## Instrument Slit Parameters
    """
    x: float
    y: float
    width: float
    height: float
    ranges: List[List[int]]

    def __init__(self, x: float, y: float, width: float, height: float, ranges: List[Tuple[int, int]] = []):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        frange = []
        for range in ranges:
            if len(range) != 2:
                continue
            if range[0] > range[1]:
                range = [range[1], range[0]]
            else:
                range = [range[0], range[1]]
            frange.append(range)
        self.ranges = frange


@dataclass
class MisFeatures:
    """## Instrument spectral features
    """
    wavelength: int  # wavelength (Angstrom)
    # key of slit from which light ends up to this panel
    slit_key: str
    plot_styles: Dict[str, int | float | str]  # color
    name: str

    def __init__(self, wavelength: int, slit_key: Optional[str] = None, plot_styles: Optional[dict] = None, name: Optional[str] = None):
        self.wavelength = wavelength
        if slit_key is None:
            slit_key = ''
        self.slit_key = slit_key
        if plot_styles is None:
            plot_styles = {}
        self.plot_styles = plot_styles
        if name is None:
            name = ''
        self.name = name


@dataclass
class MisGratingCfg:
    """## Instrument Adjustment Parameters
    """
    alpha: float  # angle of incidence (deg)
    gamma_ofst: float = 0  # grating incidence angle (deg)


@dataclass
class MisCamera:
    aperture: float # aperture area (mm^2)
    scale: float  # pixel scale (mm/pixel)
    pixel_size: float  # pixel size (mm)
    well_depth: float  # well depth (e-)
    exposure: float  # exposure time (s)
    # quantum efficiency curve (wavelength, qe)
    qe_curve: Optional[List[List[float], List[float]]] = None
    readout_noise: Optional[float] = None  # readout noise (e-) per pixel
    dark_current: Optional[float] = None  # dark current (e-/s/pixel)

    def __init__(self, aperture: float, scale: float, pixel_size: float, well_depth: float, exposure: float, qe_curve: Optional[List[List[float], List[float]]] = None, readout_noise: Optional[float] = None, dark_current: Optional[float] = None):
        self.aperture = aperture
        self.scale = scale
        self.pixel_size = pixel_size
        self.well_depth = well_depth
        self.exposure = exposure
        if qe_curve is not None:
            if len(qe_curve) != 2:
                raise ValueError('Quantum efficiency curve should be a list of two lists.')
            if len(qe_curve[0]) != len(qe_curve[1]):
                raise ValueError('Quantum efficiency curve should have the same length of wavelength and qe.')
        self.qe_curve = qe_curve
        self.readout_noise = readout_noise
        self.dark_current = dark_current


@dataclass
class MisInstrument:
    """## Instrument Parameters
    """
    system: str  # Instrument name
    optics: MisGrating  # Instrument Optics Parameters
    # Instrument Adjustment Parameters
    instrument: Optional[MisGratingCfg] = None
    # Instrument interest wavelength parameters
    lines: List[MisFeatures] = None
    camera: Optional[MisCamera] = None

    def min_lambda(self) -> int:
        """## Minimum Wavelength

        ### Returns:
            - `int`: minimum wavelength in Angstrom
        """
        if not self.lines:
            return -np.inf
        lams = [self.lines[k].wavelength for k in self.lines]
        return min(lams)

    def max_lambda(self) -> int:
        """## Maximum Wavelength

        ### Returns:
            - `int`: maximum wavelength in Angstrom
        """
        if not self.lines:
            return np.inf
        lams = [self.lines[k].wavelength for k in self.lines]
        return max(lams)

    def lambdas(self) -> List[int]:
        """## Wavelengths

        ### Returns:
            - `List[int]`: list of wavelengths in Angstrom
        """
        if not self.lines:
            return []
        return [self.lines[k].wavelength for k in self.lines]
# %%
