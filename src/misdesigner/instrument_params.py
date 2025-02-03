# %%
from __future__ import annotations
from typing import Any, Dict, List, Optional, SupportsFloat as Numeric, Tuple
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from matplotlib.axes import Axes
import numpy as np


@dataclass_json
@dataclass
class MisConfig:
    """## Instrument Optics Parameters
    """
    fl_collimator: float  # collimator focal length (mm)
    fl_mosaic: float  # grating camera focal length (mm)
    sigma: float  # groove density (grooves/mm)
    # slit X center, Y center, width, height
    slits: Dict[str, MisSlit]
    # mosaic X center, Y center, width, height
    mosaic: MisMosaic


@dataclass_json
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
    windows: Optional[List[MisMosaicFilter]] = None

    def __init__(self, x: float, y: float, width: float, height: float, windows: Optional[List[MisMosaicFilter]] = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.windows = windows

    def min_lambda(self) -> int:
        """## Minimum Wavelength

        ### Returns:
            - `int`: minimum wavelength in Angstrom
        """
        if not self.windows:
            return -np.inf
        lams = [window.min_lambda() for window in self.windows]
        return min(lams)

    def max_lambda(self) -> int:
        """## Maximum Wavelength

        ### Returns:
            - `int`: maximum wavelength in Angstrom
        """
        if not self.windows:
            return np.inf
        lams = [window.max_lambda() for window in self.windows]
        return max(lams)

    def get_xrange(self, window_name: str) -> Optional[Tuple[float, float]]:
        if not self.windows:
            return None
        for window in self.windows:
            if window.name == window_name:
                return window.get_xrange()
        return None

    def get_yrange(self, window_name: str) -> Optional[Tuple[float, float]]:
        if not self.windows:
            return None
        for window in self.windows:
            if window.name == window_name:
                return window.get_yrange()
        return None

    def get_window(self, window_name: str) -> Optional[MisMosaicFilter]:
        if not self.windows:
            return None
        for window in self.windows:
            if window.name == window_name:
                return window
        return None

    def get_window_names(self) -> List[str]:
        if not self.windows:
            return []
        return [window.name for window in self.windows]


@dataclass_json
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
    ranges: List[Tuple[float, float]]
    name: str

    def __init__(self, x: float, y: float, width: float, height: float, ranges: List[Tuple[float, float]], name: str):
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
                range = (range[1], range[0])
            else:
                range = (range[0], range[1])
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

    def min_lambda(self) -> int:
        """## Minimum Wavelength

        ### Returns:
            - `int`: minimum wavelength in Angstrom
        """
        if not self.ranges:
            return -np.inf
        lams = [range[0] for range in self.ranges]
        return min(lams)

    def max_lambda(self) -> int:
        """## Maximum Wavelength

        ### Returns:
            - `int`: maximum wavelength in Angstrom
        """
        if not self.ranges:
            return np.inf
        lams = [range[1] for range in self.ranges]
        return max(lams)


@dataclass_json
@dataclass
class MisSlit:
    """## Instrument Slit Parameters
    """
    x: float
    y: float
    width: float
    height: float
    ranges: Optional[List[Tuple[float, float]]] = None

    def __init__(self, x: float, y: float, width: float, height: float, ranges: List[Tuple[float, float]] = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        if ranges is not None:
            frange = []
            for range in ranges:
                if len(range) != 2:
                    continue
                if range[0] > range[1]:
                    range = (range[1], range[0])
                else:
                    range = (range[0], range[1])
                frange.append(range)
        else:
            frange = None
        self.ranges = frange

    def min_lambda(self) -> int:
        """## Minimum Wavelength

        ### Returns:
            - `int`: minimum wavelength in Angstrom
        """
        if not self.ranges:
            return -np.inf
        lams = [range[0] for range in self.ranges]
        return min(lams)

    def max_lambda(self) -> int:
        """## Maximum Wavelength

        ### Returns:
            - `int`: maximum wavelength in Angstrom
        """
        if not self.ranges:
            return np.inf
        lams = [range[1] for range in self.ranges]
        return max(lams)


@dataclass_json
@dataclass
class MisFeatures:
    """## Instrument spectral features
    """
    wavelength: int  # wavelength (Angstrom)
    # key of slit from which light ends up to this panel
    slit_key: Optional[str] = None
    plot_styles: Optional[Dict[str, Any]] = None  # color
    name: Optional[str] = None  # name of the feature


@dataclass_json
@dataclass
class MisGratingCfg:
    """## Instrument Adjustment Parameters
    """
    alpha: float  # angle of incidence (deg)
    gamma_ofst: float = 0  # grating incidence angle (deg)


@dataclass_json
@dataclass
class MisCamera:
    aperture: float  # aperture area (mm^2)
    scale: float  # pixel scale (mm/pixel)
    pixel_size: float  # pixel size (mm)
    exposure: float  # exposure time (s)
    optical_efficiency: float = 1  # optical efficiency
    # image width (pixels), if specified, the mosaic is assumed to be these many pixels long
    width: Optional[int] = None
    # image height (pixels), if specified, the mosaic is assumed to be these many pixels tall
    height: Optional[int] = None
    well_depth: Optional[float] = None  # well depth (e-)
    # quantum efficiency curve (wavelength, qe)
    qe_curve: Optional[Tuple[List[float], List[float]]] = None
    readout_noise: Optional[float] = None  # readout noise (e-) per pixel
    dark_current: Optional[float] = None  # dark current (e-/s/pixel)

    def __init__(self, aperture: float, scale: float, pixel_size: float, exposure: float, optical_efficiency: float = 1, well_depth: Optional[float] = None, qe_curve: Optional[Tuple[List[float], List[float]]] = None, readout_noise: Optional[float] = None, dark_current: Optional[float] = None, width: Optional[int] = None, height: Optional[int] = None):
        self.aperture = aperture
        self.scale = scale
        self.pixel_size = pixel_size
        if well_depth is not None and well_depth < 0:
            raise ValueError('Well depth should be positive.')
        self.well_depth = well_depth
        self.exposure = exposure
        self.optical_efficiency = optical_efficiency
        if qe_curve is not None:
            if len(qe_curve) != 2:
                raise ValueError(
                    'Quantum efficiency curve should be a list of two lists.')
            if len(qe_curve[0]) != len(qe_curve[1]):
                raise ValueError(
                    'Quantum efficiency curve should have the same length of wavelength and qe.')
        self.qe_curve = qe_curve
        self.readout_noise = readout_noise
        self.dark_current = dark_current
        self.width = width
        self.height = height


@dataclass_json
@dataclass
class MisInstrument:
    """## Instrument Parameters
    """
    system: str  # Instrument name
    optics: MisConfig  # Instrument Optics Parameters
    # Instrument Adjustment Parameters
    alignment: Optional[MisGratingCfg] = None
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
