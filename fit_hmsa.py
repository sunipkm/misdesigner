# %%
from __future__ import annotations
from glob import glob
import os
import sys
from time import perf_counter_ns
from typing import Any, Dict, List, Tuple
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
import matplotlib as mpl
from misdesigner import *
import yaml
from dataclasses import asdict
import xarray as xr
from astropy.io import fits
from PIL import Image

usetex = False

if not usetex:
    # computer modern math text
    mpl.rcParams.update({'mathtext.fontset': 'cm'})
mpl.rc('font', **{'family': 'serif',
       'serif': ['Times' if usetex else 'Times New Roman']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=usetex)
# %%


class Nearest:
    def __init__(self, dir='./'):
        self._dir = dir

    def __call__(self, name):
        files = natsorted(glob(os.path.join(self._dir, '*.fits')))
        for file in files:
            if name in file:
                return file
        return None


watch = Nearest()

# %%
SYSTEM = 'HMS-A ORIGIN'
SLIT_WIDTH = 50
SLIT_TO_EDGE = -5.93  # -3.55 -> -70.8 (α)
EDGE_TO_MOSAIC = -2.3
MOSAIC_WIDTH = 52.5
MOSAIC_HEIGHT = 58.98
MOSAIC_X0 = SLIT_TO_EDGE + EDGE_TO_MOSAIC
MOSAIC_X1 = MOSAIC_X0 - MOSAIC_WIDTH
slit_height = 66.5
RATIO = 400 / 443
FL1 = 416
FL2 = 421
ALPHA = 71.12
# FL1 = 420
# FL2 = 430
# ALPHA = 71.08
GAMMA = -0.25
grat = MisConfig(FL1, FL2, 98.76,
                 {
                     # (7150, 10950)]),
                     'BL': MisSlit(29, -slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(3150, 4450)]),
                     'BR': MisSlit(0, -slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(4950, np.inf)]),
                     'TL': MisSlit(29, slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(5900, np.inf)]),
                     'TR': MisSlit(0, slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(4350, 5000)]),
                 },
                 MisMosaic((MOSAIC_X0 + MOSAIC_X1)*0.5, (-1.8+2.5),
                           MOSAIC_WIDTH,  MOSAIC_HEIGHT,
                           [
                     # Hβ, 20nm around 4861
                     MisMosaicFilter(0, 1.8, 24.63, 26.96, [
                         (4860-50, 4861+50)], name='Hβ'),
                     # Hα, 20nm around 6563
                     MisMosaicFilter(
                         24.63, 1.8, 27.57 - 2.68, 26.96, [(6500-125, 6563+125)], name='Hα'),
                     # OI, 10nm around 7774
                     MisMosaicFilter(
                         0, 26.96 + 1.8, 8.13, 27.72, [(4300-50, 4300+50)], name='4278'),
                     # OI, 10nm around 5580
                    #  MisMosaicFilter(
                    #      8.13, 26.96 + 1.8, 8, 27.72, [(5500-125, 5500+125)], name='5577'),
                    MisMosaicFilter(
                         8.13, 26.96 + 1.8, 8, 27.72, [(5500, 5600)], name='5577'),
                     # OI, 10nm around 6300
                     MisMosaicFilter(
                         8.13 + 8, 26.96 + 1.8, 7.5, 27.72, [(6320-50, 6320+50)], name='6300'),
                     MisMosaicFilter(8.13 + 8 + 7.5, 26.96 + 1.8, 28.57 - 2.68, 27.72, [
                         # N2+, 10nm around 4300
                         (7750-125, 7750+125)], name='7774'),
                 ]))
model = InstrumentModel(SYSTEM, grat, gamma_ofst=GAMMA, alpha=-ALPHA)
fig, ax = model.plot_lines([
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568, 7808, 7860
],
    fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
SCALE = 65.25
qe = np.loadtxt('imx533_rel_qe.csv', delimiter=',').T
qe[0] *= 10  # nm -> Angstrom
qe[1] *= 0.91  # max QE 91%
camera = MisCamera(np.pi*9**2,
                   1,
                   1/SCALE,
                   1,
                   qe_curve=(qe[0], qe[1]))
model.set_camera(camera)
mmap = model.mosaic_map(unique=True)

# data = fits.open(watch('1650'))[1].data
# img = Image.fromarray(data)
# img.save('hmsa_origin_sky.png')
# img = Image.open('hmsa_origin_night.png')
img = Image.open('hmsa_origin_sky_fixed.png')
img = img.rotate(-.411)
img = img.transpose(Image.FLIP_LEFT_RIGHT)
# img = img.transpose(Image.FLIP_TOP_BOTTOM)
(width, height) = mmap.wavelength.shape[::-1]
fullimg = Image.new('I;16', mmap.wavelength.shape[::-1], color=0)
allimg = Image.new('I;16', mmap.wavelength.shape[::-1], color=0)
nwidth = width / SCALE
nheight = height / SCALE
fullimg.paste(img, (110, 410))
allimg.paste(fullimg.crop((0, 0, width, height/2)), (0, 0))
# fullimg = np.asarray(fullimg)
# vmin = np.percentile(fullimg, 1)
# vmax = np.percentile(fullimg, 99.5)
# ax.imshow(np.asarray(fullimg)[0:height//2],
#           origin='lower', extent=(0, -nwidth, 0, nheight/2),
#           aspect='equal',
#           #   vmin=vmin, vmax=vmax,
#           zorder=0)
img = Image.open('hmsa_origin_night.png')
img = img.rotate(-.411)
img = img.transpose(Image.FLIP_LEFT_RIGHT)
# img = img.transpose(Image.FLIP_TOP_BOTTOM)
SCALE = 65.25
fullimg = Image.new('I;16', (width, height), color=0)
fullimg.paste(img, (110, 410))
allimg.paste(fullimg.crop((0, height/2, width, height)), (0, height//2))
# fullimg = np.asarray(fullimg)
# vmin = np.percentile(fullimg, 1)
# vmax = np.percentile(fullimg, 99.5)
# ax.imshow(np.asarray(fullimg)[height//2:],
#           origin='lower', extent=(0, -nwidth, nheight/2, nheight),
#           aspect='equal',
#           #   vmin=vmin, vmax=vmax,
#           zorder=0)
ax.imshow(np.asarray(allimg),
          origin='lower', extent=(0, -nwidth, 0, nheight),
          aspect='equal',
          #   vmin=vmin, vmax=vmax,
          zorder=0)
# plt.savefig('hmsa_origin_ref.png', dpi=2400, bbox_inches='tight', transparent=True)
plt.show()
# model.intensity_model(source_wl, source_int, camera)
# %%
model.store('hmsa_origin_ship.json', True)
print('Saved instrument model.')
print('Finished.\n')
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
        self._mmap = None
        self._imaps: List[Tuple[str, MisMosaicFilter, np.ndarray, Dict[str, np.ndarray]]] = None
        self._setup()

    def _beta_from_lam(self, prodgrid: np.ndarray, gammagrid: np.ndarray, slit: Any) -> np.ndarray:
        model = self._model
        alpha = model._relative_alpha(slit, model._alpha) # model alpha is the grating angle
        alpha = np.sin(np.deg2rad(alpha))
        gamma = model._gamma_from_mosaic(gammagrid) # output angle
        gamma = model._gamma_from_image(gamma) # input angle, grating coordinates
        gamma = np.sin(np.deg2rad(gamma))
        beta = prodgrid / (gamma * model._den) - alpha
        beta = np.rad2deg(np.arcsin(beta)) # beta in grating coordinates
        beta = model._beta_to_image(beta, model._alpha) # beta in image coordinates
        beta = model._beta_to_mosaic(beta) # beta in mosaic coordinates
        return beta

    def _setup(self):
        mmap = self._model.mosaic_map(unique=True, report=True) # map of wavelength in the mosaic coordinate system
        self._mmap = mmap
        imaps = [] 
        for sname in mmap['slit']:
            source = mmap['source'].sel(slit=sname).drop('slit') # select all betta, gamma that is illuminated by this slit
            if not np.any(source.values):
                continue # this slit does not illuminate anything, move on
            for window in self._model.mosaic.windows: # for each window in the mosaic
                xran = window.get_xrange() # beta range
                yran = window.get_yrange() # gamma range
                smap = mmap.where(source==True, drop=True) # select the part of the mosaic that is illuminated by this slit
                smap = smap.sel(gamma=slice(*yran), beta=slice(*xran)) # select the part of the mosaic that is in the window
                wl = smap['wavelength'] # get the wavelengths
                if wl.values.size == 0: # no wavelengths in this window, move on
                    continue
                order = np.unique(smap['order']) # find the order
                order = order[~np.isnan(order)] # NaN filter
                if len(order) > 1: # multiple orders present
                    raise NotImplementedError(f'TODO: Handle the case where multiple orders are present: {order}')
                elif len(order) == 0: # no orders present, move on
                    continue
                order = order[0] # select the order, which is enforced to be length 1
                fig = plt.figure()
                wl.plot(figure=fig)
                fig.suptitle(f'{sname.values} - {window.name}')
                plt.show()
                wlmin = np.nanmin(wl)
                wlmax = np.nanmax(wl)
                print(wlmin, wlmax)
                minwhere = np.where(wl.values == wlmin)
                maxwhere = np.where(wl.values == wlmax)
                wlmid = np.nanmax(wl.values[minwhere[0], :])
                midwhere = np.nanargmax(wl.values[minwhere[0], :])
                wl_len = abs(midwhere - minwhere[1][0])
                midwhere = np.nanargmin(np.abs(wl.values[maxwhere[0], :] - wlmid))
                wl_len += abs(midwhere - maxwhere[1][0])
                sign = 1 if maxwhere[1] >= minwhere[1] else -1
                # this is the output wavelength grid
                wl_array = np.linspace(wlmin, wlmax, wl_len, endpoint=True)[::sign]
                print(wlmin, wlmax)
                mlam, mgam = np.meshgrid(wl_array, wl['gamma'].values) # gamma here is in mosaic
                if window.name == '6300':
                    plt.imshow(mlam)
                    plt.colorbar()
                    plt.show()
                mbet = self._beta_from_lam(mlam*order, mgam, str(sname.values)) # beta mesh grid
                if window.name == '6300':
                    plt.imshow(mbet)
                    plt.colorbar()
                    plt.show()
                xform = np.zeros((2, *(mbet.shape)), dtype=float) # reverse map
                bmin, bmax = np.nanmin(wl.beta.values), np.nanmax(wl.beta.values)
                gmin, gmax = np.nanmin(wl.gamma.values), np.nanmax(wl.gamma.values)
                mbet -= bmin
                mgam -= gmin
                mbet /= (bmax - bmin)
                mgam /= (gmax - gmin)
                xform[0, :, :] = mgam * len(wl.gamma.values)
                xform[1, :, :] = mbet * len(wl.beta.values)
                coords = {'gamma': wl.gamma.values, 'lambda': wl_array}
                imaps.append((sname.values, window, xform, coords))
        self._imaps = imaps
        return imaps
    @property
    def beta_grid(self):
        return self._mmap['beta']
    
    @property
    def gamma_grid(self):
        return self._mmap['gamma']

    def straighten_image(self, image: xr.DataArray, win_name: str):
        from skimage import transform
        if self._imaps is None:
            raise ValueError('Must setup first')
        ret = []
        for slit, window, xform, coords in self._imaps:
            if window.name != win_name:
                continue
            xran = window.get_xrange()
            yran = window.get_yrange()
            data = image.sel(gamma=slice(*yran), beta=slice(*xran))
            print(xran, xform[1].min(), xform[1].max())
            print(yran, xform[0].min(), xform[0].max())
            out = transform.warp(data.values[:, ::-1], xform, cval=np.nan) # why the fuck do I need to reverse the X axis
            print(data.values.shape, out.shape)
            plt.imshow(data.values)
            plt.colorbar()
            plt.show()
            out = xr.DataArray(out, coords=coords)
            out.plot()
            plt.show()
            # plt.imshow(out, aspect='equal')
            # plt.colorbar()
            # plt.show()
            ret.append(out)
        ret = xr.concat(ret, dim='gamma')
        return ret
            

# %%
ret = MapPixel2Wavelength(model)
# %%
img = xr.DataArray(np.asarray(allimg).astype(float), 
                   dims=['gamma', 'beta'],
                   coords={'gamma': ret.gamma_grid, 'beta': ret.beta_grid})
# %%
for slit, window, r in ret:
    plt.suptitle(f'{slit} - {window}')
    plt.imshow(r)
    plt.colorbar()
    plt.show()
#%%

#%%
slit,window,r = ret[0]
ulam = np.unique(r) # unique wavelengths in the full window
target_lam = np.linspace(np.min(ulam),np.max(ulam),len(ulam)) #linear target lambda

# get valid gamma for the window
#make a meshgrid of target_lam and gamma 
# plug those into beta_from_lam to get beta of straighted lam
# make the reverse map of size (2,gamma, beta)
    # rmap[0,:,:] = gamma grid
    # rmap[1,:,:] = beta grid
# plug into wrap
# plug the straighted lam and beta into the reverse map to get the gamma

# mlam,mgam
# %%
xlim = model.mosaic.windows[0].get_xrange()
ylim = model.mosaic.windows[0].get_yrange()
wl = mmap['wavelength'].sel(gamma=slice(*ylim), beta=slice(*xlim))
wlmin = np.nanmin(wl)
wlmax = np.nanmax(wl)
# %%
np.where(wl.values == wlmin)
# %%
wlmid = np.nanmax(wl.values[0, :])
# %%
np.where(wl.values == wlmax)

# %%
np.argmin(np.abs(wl.values[1641, :] - wlmid))
# %%
ds1 = xr.DataArray(np.ones((10, 20)), dims=['x', 'y'], coords={'x': np.arange(10), 'y': np.arange(20)})
ds2 = xr.DataArray(np.full((10, 20), 2), dims=['x', 'y'], coords={'x': np.arange(20, 30), 'y': np.arange(50, 70)})
concat = xr.concat([ds1, ds2], dim='x')
# %%
