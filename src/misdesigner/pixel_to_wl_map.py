# %%
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args
from natsort import natsorted
from skimage import transform
import numpy as np
from xarray import DataArray, Dataset, MergeError, concat
import warnings

from .instrument_params import MisMosaicFilter
from .instrument_model import MisInstrumentModel
# %%

StraightenCoordinate = Literal['Mosaic', 'Slit', 'Grating']


class MisCurveRemover:
    """## This class is used to produce wavelength maps from images obtained with a MISDesigned instrument.
    """

    def __init__(self, model: MisInstrumentModel, uniquemap: Optional[Dataset] = None, *, report: bool = False):
        """## Constructor for the SpectraCurveRemover class.

        ### Args:
            - `model (MisInstrumentModel)`: The model of the instrument.
            - `uniquemap (Optional[Dataset], optional)`: Wavelength map of the mosaic, generated with `model.mosaic_map(unique=True)`. Defaults to None, where it will be generated.

        ### Raises:
            - `ValueError`: Model must have a camera
            - `ValueError`: Model must have a mosaic
            - `ValueError`: Model must have mosaic windows
        """
        if model._camera is None:
            raise ValueError('Model must have a camera')
        if model.mosaic is None:
            raise ValueError('Model must have a mosaic')
        if model.mosaic.windows is None or len(model.mosaic.windows) == 0:
            raise ValueError('Model must have mosaic windows')
        self._model = model
        self._mmap = uniquemap
        self._imaps: List[Tuple[str, MisMosaicFilter,
                                np.ndarray,
                                Dict[str, np.ndarray],
                                DataArray
                                ]
                          ] = None
        self._setup()

    def _beta_from_lam(self, prodgrid: np.ndarray, gammagrid: np.ndarray, slit: Any) -> np.ndarray:
        model = self._model
        # model alpha is the grating angle
        alpha = model._relative_alpha(slit, model._alpha)
        alpha = np.sin(np.deg2rad(alpha))
        gamma = model._gamma_from_mosaic(gammagrid)  # output angle
        # input angle, grating coordinates
        gamma = model._gamma_from_image(gamma)
        gamma = np.sin(np.deg2rad(gamma))
        beta = prodgrid / (gamma * model._den) - alpha
        beta = np.rad2deg(np.arcsin(beta))  # beta in grating coordinates
        # beta in image coordinates
        beta = model._beta_to_image(beta, model._alpha)
        beta = model._beta_to_mosaic(beta)  # beta in mosaic coordinates
        return beta

    def _setup(self, report: bool = False):
        if self._mmap is None:
            # map of wavelength in the mosaic coordinate system
            mmap = self._model.mosaic_map(unique=True, report=report)
            self._mmap = mmap
        else:
            mmap = self._mmap
        imaps = []
        windows = []
        for sname in mmap['slit']:
            # select all betta, gamma that is illuminated by this slit
            source = mmap['source'].sel(slit=sname).drop_vars('slit')
            if not np.any(source.values):
                continue  # this slit does not illuminate anything, move on
            for window in self._model.mosaic.windows:  # for each window in the mosaic
                xran = window.get_xrange()  # beta range
                yran = window.get_yrange()  # gamma range
                # select the part of the mosaic that is illuminated by this slit
                smap = mmap.where(source == True, drop=True)
                # select the part of the mosaic that is in the window
                smap = smap.sel(gamma=slice(*yran), beta=slice(*xran))
                wl = smap['wavelength']  # get the wavelengths
                res = smap['resolution']  # get the resolution
                if wl.values.size == 0:  # no wavelengths in this window, move on
                    continue
                order = np.unique(smap['order'])  # find the order
                order = order[~np.isnan(order)]  # NaN filter
                if len(order) > 1:  # multiple orders present
                    warnings.warn(
                        f'TODO: Handle the case where multiple orders are present: {order}')
                elif len(order) == 0:  # no orders present, move on
                    continue
                # select the order, which is enforced to be length 1
                order = order[0]
                wlmin = np.nanmin(wl)
                wlmax = np.nanmax(wl)
                minwhere = np.where(wl.values == wlmin)
                maxwhere = np.where(wl.values == wlmax)
                wlmid = np.nanmax(wl.values[minwhere[0], :])
                midwhere = np.nanargmax(wl.values[minwhere[0], :])
                wl_len = abs(midwhere - minwhere[1][0])
                midwhere = np.nanargmin(
                    np.abs(wl.values[maxwhere[0], :] - wlmid))
                wl_len += abs(midwhere - maxwhere[1][0])
                sign = 1 if maxwhere[1] >= minwhere[1] else -1
                # this is the output wavelength grid
                wl_array = np.linspace(
                    wlmin, wlmax, wl_len, endpoint=True)[::sign]
                # gamma here is in mosaic
                mlam, mgam = np.meshgrid(wl_array, wl['gamma'].values)
                mbet = self._beta_from_lam(
                    mlam*order, mgam, str(sname.values))  # beta mesh grid
                xform = np.zeros((2, *(mbet.shape)),
                                 dtype=float)  # reverse map
                bmin, bmax = np.nanmin(
                    wl.beta.values), np.nanmax(wl.beta.values)
                gmin, gmax = np.nanmin(
                    wl.gamma.values), np.nanmax(wl.gamma.values)
                mbet -= bmin
                mgam -= gmin
                mbet /= (bmax - bmin)
                mgam /= (gmax - gmin)
                xform[0, :, :] = mgam * len(wl.gamma.values)
                xform[1, :, :] = mbet * len(wl.beta.values)
                coords = {
                    'gamma': wl.gamma.values,
                    'lambda': wl_array,
                    'beta': wl.beta.values
                }
                imaps.append((sname.values, window, xform, coords, wl / res))
                windows.append(window.name)
        self._imaps = imaps
        self._windows = natsorted(tuple(set(windows)))

    @property
    def beta_grid(self) -> DataArray:
        """## Get the beta grid of the wavelength map in mosaic coordinates.

        ### Returns:
            - `DataArry`: The beta grid of the wavelength map.
        """
        return self._mmap['beta']

    @property
    def gamma_grid(self) -> DataArray:
        """## Get the gamma grid of the wavelength map in mosaic coordinates.

        ### Returns:
            - `DataArry`: The gamma grid of the wavelength map.
        """
        return self._mmap['gamma']

    @property
    def windows(self) -> Tuple[str]:
        """## Get the names of the windows in the wavelength map.

        ### Returns:
            - `Tuple[str]`: The names of the windows in the wavelength map.
        """
        return self._windows

    def straighten_image(self, image: DataArray, win_name: str, *, inplace: bool = True, coord: StraightenCoordinate = 'Mosaic') -> DataArray:
        """## Straighten an image using the wavelength map.

        ### Args:
            - `image (DataArray)`: Input image to be straightened. Must be in the same coordinate system as the wavelength map (mosaic).
            - `win_name (str)`: The name of the window to use for straightening.
            - `inplace (bool, optional)`: If True, the input image will be modified in place. Defaults to True.
            - `coord (StraightenCoordinate, optional): If `Mosaic`, use the mosaic coordinate (mm) for the coordinate parallel to grating ruling. If `Grating`, use the grating coordinate (angle, deg) for light incident on the grating. If `Slit`, use the instrument coordinate (mm) at the slit.

        ### Returns:
            - `DataArray`: The straightened image.
            Note: The straightened image is in the ('gamma', 'lambda') coordinate system.
            Lambda is the wavelength in nanometers. Image intensity is returned per nanometers.
        """
        if self._imaps is None:
            raise ValueError('Must setup first')
        ret: List = []

        for _, window, xform, coords, res in self._imaps:
            if window.name != win_name:
                continue
            xran = (coords['beta'].max(), coords['beta'].min())
            yran = (coords['gamma'].min(), coords['gamma'].max())
            data = image.sel(gamma=slice(*yran), beta=slice(*xran))
            if inplace:
                try:
                    data /= res
                except (MergeError, ValueError):
                    data = data / res
            else:
                data = data / res
            # why the fuck do I need to reverse the X axis
            out = transform.warp(data.values[:, ::-1], xform, cval=np.nan)
            if coord == 'Mosaic':
                gamma = ('gamma', coords['gamma'],
                         {
                    'coodinate': coord,
                    'unit': 'mm',
                    'description': 'Height in the mosaic coordinate, increasing from the bottom.'
                })
            elif coord == 'Grating':
                gamma = ('gamma', self._model._gamma_from_image(self._model._gamma_from_mosaic(coords['gamma'])),
                         {
                    'coodinate': coord,
                    'unit': 'deg',
                    'description': 'Angle in the instrument coordinate.'
                })
            elif coord == 'Slit':
                gamma = ('gamma', self._model._gamma_to_slit(self._model._gamma_from_image(self._model._gamma_from_mosaic(coords['gamma']))),
                         {
                    'coodinate': coord,
                    'unit': 'mm',
                    'description': 'Height in the instrument coordinate.'
                })
            else:
                raise ValueError(
                    f'Invalid value for coord: {coord}. Valid values are {", ".join(get_args(StraightenCoordinate))}.')
            out = DataArray(out*10, coords={
                'gamma': gamma,
                'wavelength': ('wavelength', coords['lambda']/10,
                               {
                               'unit': 'nm',
                               'description': 'Wavelength in nanometer',
                               }),
            })
            ret.append(out)
        out: DataArray = concat(ret, dim='gamma')
        out = out.sortby('gamma')
        out = out.sortby('wavelength')
        if image.attrs.get('unit') is not None:
            out.attrs['unit'] = image.attrs['unit'] + ' nm^{-1}'
        else:
            out.attrs['unit'] = 'nm^{-1}'
        return out
