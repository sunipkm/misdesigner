from .line_predictor import LinePredictor, PlotMode
# from .pixel_to_wl_map import MapPixel2Wl
from .instrument_params import *
import importlib.metadata as metadata

__version__ = metadata.version('misdesigner')

__all__ = ['LinePredictor',
           'PlotMode',
           'MisGrating',
           'MisMosaic',
           'MisMosaicFilter',
           'MisSlit',
           'MisFeatures',
           'MisGratingCfg',
           'MisInstrument',
           'MisCamera',
           '__version__',]
