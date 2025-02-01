# %%
from __future__ import annotations
from typing import List, Optional, SupportsFloat as Numeric, Tuple
import os
from skimage import transform
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

from xarray import DataArray

from .instrument_params import MisMosaicFilter
from .instrument_model import InstrumentModel
from .utils import find_nearest, open_fits
# from .instrument_model import HMS_ImagePredictor
# %%


class MapPixel2Wavelength:
    def __init__(self, model: InstrumentModel):
        if model._camera is None:
            raise ValueError('Model must have a camera')
        if model.mosaic is None:
            raise ValueError('Model must have a mosaic')
        if model.mosaic.windows is None or len(model.mosaic.windows) == 0:
            raise ValueError('Model must have mosaic windows')
        self._model = model

    def _setup(self):
        mmap = self._model.mosaic_map(unique=True)
        imaps = []
        for window in self._model.mosaic.windows:
            xr = window.get_xrange()
            yr = window.get_yrange()
            wl = self._mmap['wavelength'].sel(gamma=slice(*yr), beta=slice(*xr))
            wlmin = np.nanmin(wl)
            wlmax = np.nanmax(wl)
            ...

    @property
    def height(self) -> int:
        return len(self._mmap['gamma'])

    @property
    def width(self) -> int:
        return len(self._mmap['beta'])

    def straighten_image(self, image: np.ndarray, panel: Optional[str] = None) -> List[Tuple[MisMosaicFilter, np.ndarray]]:
        if image.shape != (self.height, self.width):
            raise ValueError(
                'Image must have the same dimensions as the mosaic map')
        image: DataArray = DataArray(
            image, dims=['gamma', 'beta'], coords={'gamma': self._mmap['gamma'], 'beta': self._mmap['beta']})
        straightened_images = []
        # filtering by panel name
        if panel is not None:
            panel_names = [window.name for window in self._model.mosaic.windows]
            panel_names = list(filter(None, panel_names))
            if len(panel_names) != len(self._model.mosaic.windows):
                raise ValueError('All mosaic windows must have names for filtering')
        # make the inverse maps
        imaps = []
        for window in self._model.mosaic.windows:
            # if panel is not None and panel not in window.name:
            #     continue
            xr = window.get_xrange()
            yr = window.get_yrange()
            wl = self._mmap['wavelength'].sel(gamma=slice(*yr), beta=slice(*xr))
            wlmin = np.nanmin(wl)
            wlmax = np.nanmax(wl)
            ...
        return straightened_images

    pass


class MapPixel2Wl:
    """This Class maps the each pixel of the detector to the corresponding wavelegth that is indicent on it though the hms instrument.
    """

    def __init__(self, predictor: HMS_ImagePredictor) -> None:
        self.ip = predictor  # HMs image predictor
        self.hmsParamDict = self.ip.hmsParamDict
        self.wlParamDict = self.ip.wlParamDict
        self.sigma = predictor.sigma

        print('Calculating Gamma...')
        self.gammagrid = self.get_gamma_grid()
        print('Calculating Beta...')
        self.betagrid = self.get_beta_grid()
        print('Beta grid shape:', self.betagrid.shape)
        print('Calculating Panel...')
        self.panelgrid = self.get_value_grid('wl')
        print('Calculating Alpha...')
        self.alphagrid = self.get_value_grid('alpha')
        print('Calculating Order of diffraction...')
        self.ordergrid = self.get_value_grid('diffractionOrder')
        print('Calculating the transforms...', end=' ')
        sys.stdout.flush()
        self.lambdagrid = None
        wavelengths = [int(x) for x in self.wlParamDict.keys()]
        wavelengths.sort()
        xforms = {}
        for wl in wavelengths:
            print(wl, end='.. ')
            sys.stdout.flush()
            # Chose the target wl array closest to beta = 90 deg to straighted img against.
            gamma_grid = self.extract_wlpanel(wl, self.gammagrid)
            gidx, _ = find_nearest(gamma_grid[:, 50], self.ip.mgammadeg)
            target_beta = self.extract_wlpanel(wl, self.betagrid)
            target_beta = target_beta[gidx, :]
            alpha = self.extract_wlpanel(wl, self.alphagrid)[0, 0]
            order = self.extract_wlpanel(wl, self.ordergrid)[0, 0]
            bmin = target_beta.min()
            bmax = target_beta.max()
            target_lam = self.calc_lamda_gratingeqn(
                alpha, np.array([bmin, bmax]), self.ip.mgammadeg, order)
            # target lambda
            target_lam = np.linspace(
                target_lam.min(), target_lam.max(), len(target_beta))
            # gamma, lambda grid
            mlam, mgam = np.meshgrid(target_lam, gamma_grid[:, 0])
            # gamma, beta grid
            mbeta = self.calc_beta_gratingeqn(alpha, mlam, mgam, order)
            # beta, in pix
            mbeta -= target_beta.min()
            mbeta /= target_beta.max() - target_beta.min()
            mbeta *= len(target_beta)
            # gamma, in pix
            mgam -= gamma_grid.min()
            mgam /= gamma_grid.max() - gamma_grid.min()
            mgam *= len(gamma_grid[:, 0])
            # new array
            xform = np.zeros((2, *(mgam.shape)), dtype=float)
            xform[0, :, :] = mgam
            xform[1, :, :] = mbeta
            xforms[wl] = (target_lam, xform)
        self.xforms = xforms
        print('Done.')

    def calc_beta_gratingeqn(self, alpha: Numeric, lam: Numeric, gamma: Numeric, order: int) -> Numeric:
        """Calulates β using the grating equation \n λ = σ(sinγ)/m * (sinα + sinβ) 
        Args:
            alpha (float): angle of incidence perpendicluar to groves, α [Degrees (0-360).
            lam (float): Wavelength, λ.
            gamma (float): angle of incidence parallel to groves,γ [Degrees (0-360)]. 
            order (int): diffraction order, m.

        Returns:
            float: Wavelength, λ.
        """
        return np.arcsin((order * lam / self.sigma / np.sin(gamma * np.pi / 180)) - np.sin(alpha * np.pi / 180)) * 180 / np.pi

    def calc_lamda_gratingeqn(self, alpha: Numeric, beta: Numeric, gamma: Numeric, order: int) -> Numeric:
        """Calulates λ using the grating equation \n λ = σ(sinγ)/m * (sinα + sinβ) 
        Args:
            alpha (float): angle of incidence perpendicluar to groves, α [Degrees (0-360)].
            beta (float): angle of diffraction, β [Degrees (0-360)].
            gamma (float): angle of incidence parallel to groves,γ [Degrees (0-360)]. 
            order (int): diffraction order, m.

        Returns:
            float: Wavelength, λ.
        """
        # alpha = correct_unit_of_angle(alpha, "rad")  # noqa: F405
        # beta = correct_unit_of_angle(beta, "rad")
        # gamma = correct_unit_of_angle(gamma, "rad")

        return (self.sigma/order)*np.sin(gamma*np.pi/180)*(np.sin(alpha*np.pi/180)+np.sin(beta*np.pi/180))

    def get_gamma_grid(self) -> np.ndarray:
        """ calculates angle of incidence parallel to grooves for all pixel postions.

        Returns:
            np.array: gamma_grid [degrees], shape = (totalpix X totalpix)
        """
        gdeg = 90 + self.ip.mm2deg(self.ip.MosaicWindowHeightmm/2, self.ip.fprime) * np.linspace(-1, 1,
                                                                                                 2*101)  # linspace of gammas that are allowed through the mosaic window, one column of gammas
        gdeg_binned = [
            list(np.linspace(np.min(gdeg), np.max(gdeg), self.ip.pix))]
        gammadeg_grid = np.array(gdeg_binned*self.ip.pix)  # grid
        return np.array(gammadeg_grid.T, dtype=float)

    def get_beta_grid(self) -> np.ndarray:
        """calculates angle of diffraction for all pixel postions.

        Returns:
            np.array:  beta_grid [degrees], shape = (totalpix X totalpix)
        """
        betafaredge = self.ip.alpha - \
            self.ip.mm2deg(
                self.hmsParamDict.mosaic_far_edge, self.ip.fprime)
        betanearedge = betafaredge + \
            self.ip.mm2deg(self.ip.MosaicWindowWidthmm, self.ip.fprime)
        # linspace of betas that are allowed through the mosaic, one row of betas
        betadeg = [list(np.linspace(betafaredge, betanearedge, self.ip.pix))]
        betadeg_grid = np.array(betadeg*self.ip.pix)  # grid
        return np.array(betadeg_grid, dtype=float)

    def get_value_grid(self, value: str) -> np.ndarray:
        """calculates value for requested variable for all pixel postions. variables can be one of the following: 'alpha','wl','diffractionorder'

        Args:
            value (str): Requestion variable can be: \n 'alpha' - incidence angle perpendicular to grooves (degrees). \n 'wl' - the wl that the panel belongs to (Angstroms). \n 'diffractionorder' - order of diffraction m.

        Raises:
            ValueError: Value must be 'Alpha', 'wl', 'diffractionOrder'. 

        Returns:
            np.array: grid of values of shape = (totalpix X totalpix)
        """
        value = value.lower()
        if value not in ['alpha', 'wl', 'diffractionorder']:
            raise ValueError(
                "Value must be 'Alpha', 'wl', 'diffractionOrder. ")
        gammagrid = self.get_gamma_grid()  # get grid of gamma values, deg.
        betagrid = self.get_beta_grid()  # get grid of beta values, deg.

        # create a grid of zeros.
        value_grid = np.zeros((self.ip.pix, self.ip.pix))

        # get the list of filter wls
        for fidx, wllist in enumerate(self.hmsParamDict.mosaic_filters):
            if fidx == 0:
                rowidx = np.where(gammagrid[:, 0] >= 90)[
                    0]  # bottom panel on image
            elif fidx == 1:
                rowidx = np.where(gammagrid[:, 0] <= 90)[
                    0]  # top panel on image
            for wlidx, wl in enumerate(wllist):
                wdict = self.wlParamDict[wl]  # get wavelength dict
                morder = int(wdict.diffraction_order)

                if wdict.slit_key in [2, 4]:
                    alpha = self.ip.alpha_slitA
                elif wdict.slit_key in [1, 3]:
                    alpha = self.ip.alpha_slitB

                if wlidx == 0:
                    # set starting x value as the left edge of its the first wl in the list
                    x = betagrid[0][0]
                else:
                    x = x1  # set start x value as the right edge of previous wl
                # set the right edge of the current wl
                x1 = x + \
                    self.ip.mm2deg(wdict.panel_window_width, self.ip.fprime)
                panelmask = (betagrid[0] >= x) & (
                    betagrid[0] <= x1)  # find idx within panel
                colidx = np.where(panelmask)[0]
                X, Y = np.meshgrid(rowidx, colidx, indexing='ij')

                if value in 'alpha':
                    val = float(alpha)
                elif value in 'wl':
                    val = int(wl)
                elif value in 'diffractionorder':
                    val = morder

                value_grid[X, Y] = val
        return np.array(value_grid)

    def get_lambda_grid(self) -> list[float]:
        """ Calculate diffracted wavelength for all pixel postions.

        Returns:
            list[float]: wavelegth array of shape totalpix X totalpix
        """
        return self.calc_lamda_gratingeqn(self.alphagrid, self.betagrid, self.gammagrid, self.ordergrid)

    def get_resolution_grid(self, slitwidth: Numeric) -> list[float]:
        """Calculate spectral resolution for all pixel postions.

        Args:
            slitwidth (float):  Width of the entrance silt, [microns](1e-6).

        Returns:
            list[float]: spectral resolution array of shape totalpix X totalpix
        """
        b = np.empty_like(self.alphagrid)
        b.fill(slitwidth)
        if self.lambdagrid is None:
            self.lambdagrid = self.get_lambda_grid()
        return self.calc_resolution_gratingeqn(self.alphagrid, self.betagrid, self.gammagrid, b, self.lambdagrid)

    def get_wlpanel_idx(self, wl: int, value_grid: np.ndarray) -> tuple[np.array, np.array]:
        """ find idices corresponding to the wavelength (int, A). Used to find the pixels that correspond to single ROI or panel.

        Args:
            wl (int): Wavelength, Angstroms.
            value_grid (np.ndarray): grid of values of shape = (totalpix X totalpix). Should be used with self.panelgrid.

        Returns:
            tuple[np.array,np.array]: rows, columns
        """
        value_grid = np.array(value_grid)
        rows, cols = np.where((value_grid == wl))
        return rows, cols

    def extract_wlpanel(self, wl: int, value_grid: np.ndarray) -> np.ndarray:
        """ extracts the value_grid that corresponds to the ROI of the wavelength.

        Args:
            wl (int): Wavelength, Angstroms.
            value_grid (np.ndarray): grid of values of shape = (totalpix X totalpix).

        Returns:
            np.ndarray: ROI_value_grid of shape (n,m)
        """

        value_grid = np.array(value_grid)
        rows, cols = self.get_wlpanel_idx(wl, self.panelgrid)
        return np.array(value_grid[np.min(rows):np.max(rows)+1, np.min(cols):np.max(cols)+1])

    def straighten_img(self, wavelength: Numeric, img: np.ndarray | str, rotate_deg: Optional[Numeric] = None, plot: bool = True, plotwlaxis: bool = True) -> tuple[np.ndarray, np.ndarray, np.array]:
        """ straighten the img by panel.

        Args:
            wavelength (float): wavelength of ROI, nm.
            img (np.ndarray | str): HMS img. it can be a 2D array or a str path to .fits file. If path does not exist, it will straighten the modeled spectra.
            rotate_deg (float, optional): rotate the orginal HMS image counter clockwise from the center, degrees. Defaults to None.
            plot (bool, optional): if true, plot the input image on the right, and straighted image on the left. Defaults to True.
            plotwlaxis (bool, optional): if true, it plots wavelength on the x-axis of the straighted img. Defaults to True.

        Raises:
            ValueError: img must be a 2D array or a str.

        Returns:
            tuple[np.ndarray, np.ndarray, np.array]: straighted img, input image, wavelength-array of straighted img. 
        """
        wl = int(wavelength*10)

        if rotate_deg is None:
            rotate_deg = self.ip.img_rot

        # extract panel from curved img
        if isinstance(img, np.ndarray):
            img = transform.rotate(
                img, rotate_deg, cval=np.nan, preserve_range=True)
            curved_img_grid = self.extract_wlpanel(wl, img)
            cbarlabel = 'Intensity [ADU]'
        elif isinstance(img, str):
            if os.path.exists(img):  # Extract panel from the hms img
                img, _ = open_fits(img)
                img = transform.rotate(
                    img, rotate_deg, cval=np.nan, preserve_range=True)
                curved_img_grid = self.extract_wlpanel(wl, img)
                cbarlabel = 'Intensity [ADU]'
            else:
                raise ValueError('No image available')
        else:
            raise ValueError('img must be a 2D array or str img path.')

        # get the inverse transform
        target_wls, xform = self.xforms[wl]
        output = transform.warp(curved_img_grid, xform, cval=np.nan)
        straight_img = output

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Plot the diffracted spectral lines (curved) on the left
            im0 = axs[0].imshow(curved_img_grid, aspect='auto')
            axs[0].set_title("Diffracted Spectral Lines")
            axs[0].set_xlabel("Pixel Position X")
            axs[0].set_ylabel("Pixel Position Y")
            fig.colorbar(im0, ax=axs[0], label=cbarlabel)

            # Plot the straightened spectral lines on the right
            im1 = axs[1].imshow(straight_img, aspect='auto')
            axs[1].set_title("Straightened Spectral Lines")

            axs[1].set_ylabel("Pixel Position Y")
            fig.colorbar(im1, ax=axs[1], label=cbarlabel)
            centralwlidx = np.argmin(np.abs(target_wls-wavelength))
            axs[1].axvline(x=centralwlidx, color='white',
                           linestyle='--', linewidth=0.5)
            if plotwlaxis:
                wllabels = [f'{x:.1f}' for x in target_wls]
                stepsize = 200
                l = axs[1].set_xticks(ticks=np.arange(
                    0, len(target_wls), stepsize), labels=wllabels[::stepsize])
                axs[1].set_xlabel("Wavelength [nm]")
            else:
                axs[1].set_xlabel("Pixel Position X")

            plt.tight_layout()
            plt.show()

        return np.array(straight_img), np.array(curved_img_grid), np.array(target_wls)


# %%
# predictor = HMS_ImagePredictor(
#     '/home/charmi/Projects/hitmis_analysis/hmspython/hmspython/Utils/hmsa_aurora_slittest.json', 67.455, 50, mgammadeg=89.95, pix=3008)
# %%
# mapping = MapPixel2Wl(predictor)
# %%
# fnames = glob('../../../Origins/2024-10-22_11_40_39Z/*.fit*')
# impath = fnames[1]
# with pf.open(impath) as hdul:
#     data = hdul[0].data.astype(np.float64)

# #resize img
# IMGSIZE = 1024
# scale_factor = (IMGSIZE / data.shape[0], IMGSIZE / data.shape[1])
# data = transform.rescale(data, scale_factor, order=3)
# rotate img
# data = transform.rotate(data,-0.4)

 # %%
# img = predictor.plot_spectral_lines('Detector', True, wls=[
#                                     557.7, 630.0, 427.8, 777.4, 486.1, 656.3, 486.1], mosaic=True, measurement=True, fprime=442.7)
# vmin = np.percentile(data, 1)
# vmax = np.percentile(data, 99)
# plt.imshow(data, cmap='pink', vmin=vmin, vmax=vmax)
# plt.colorbar()

# %%
# Straighten img
# wl = 486.1
# simg, img, wlaxis = mapping.straighten_img(
#     wavelength=wl, img=data, rotate_deg=-0.2)
# %%
# plot a bigger version of straightened img
# vmin = np.nanpercentile(simg, 1)
# vmax = np.nanpercentile(simg, 99)

# fig = plt.figure(dpi=1200)
# plt.imshow(simg, vmin=vmin, vmax=vmax, aspect='auto')
# plt.axvline(find_nearest(wlaxis, wl)[0], color='white', linewidth=0.5)
# plt.axvline(300, color='white', linewidth=0.5)
# plt.colorbar()
# plt.title(f'{wl} nm')
# %%
# fdir = 'Images/hmsA_img/20240829/*.fits'
# fnames = glob(fdir)
# fnames.sort()
# print(len(fnames))
# # %%
# idx = -300


# # %%
# def bin_and_sum_image(straightened_image, wavelength_bins, row_bin_size=100, col_bin_size=5, plot=True):
#     rows, cols = straightened_image.shape
#     num_row_bins = rows // row_bin_size
#     num_col_bins = cols // col_bin_size

#     # Initialize array to hold binned and summed data
#     binned_data = np.zeros((num_row_bins, num_col_bins))

#     for i in range(num_row_bins):
#         for j in range(num_col_bins):
#             start_row = i * row_bin_size
#             end_row = start_row + row_bin_size
#             start_col = j * col_bin_size
#             end_col = start_col + col_bin_size
#             binned_data[i, j] = np.sum(straightened_image[start_row:end_row, start_col:end_col])

#     # Adjust wavelength bins to match the binned columns
#     binned_wavelength_bins = np.mean(np.reshape(wavelength_bins[:num_col_bins * col_bin_size], (num_col_bins, col_bin_size)), axis=1)

#     if plot:
#         plt.figure(figsize=(12, 6))
#         for i in range(num_row_bins):
#             plt.plot(binned_wavelength_bins, binned_data[i, :], label=f'Bin {i+1}')

#         plt.xlabel("Wavelength")
#         plt.ylabel("Summed Intensity")
#         plt.title("Line Profiles and Read Noise")
#         plt.legend()
#         plt.show()

#     return binned_data, binned_wavelength_bins

# %%
# img,wl_arr =mapping.straighten_img(fnames[idx], 630.0)
# binned_data,wl_arr = bin_and_sum_image(img, wl_arr,col_bin_size=2)
# plt.plot(wl_arr,binned_data.mean(axis = 0),linewidth = 0.7)
# plt.axvline(x = 630.0, linestyle = '--', linewidth = 0.5)
# plt.title(format_time(fnames[idx].split('ccdi_')[-1].strip('.fits')))
# # %%
# plt.figure()

# for i in range(1,5):
#     img,wl =mapping.straighten_img(fnames[idx], 630.0, plot = False)
#     binned_data,wl_arr = bin_and_sum_image(img, wl,col_bin_size=i, plot = False)

#     plt.plot(wl_arr,binned_data[7],linewidth = 0.7, label = f'binsize = {i}')
#     plt.xlim(629.5, 630.5)
# plt.axvline(x = 630.0, linestyle = '--', linewidth = 0.5)
# plt.legend(loc = 'best')


# # %%
# plt.imshow(mapping.lambdagrid)
# # %%
# img,wl_arr =mapping.straighten_img('',777.4)
# print(np.min(wl_arr), np.max(wl_arr))
# # %%

# %%

# %%
