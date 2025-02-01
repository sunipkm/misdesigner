# %%
from __future__ import annotations
from glob import glob
import os
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
import matplotlib as mpl
from misdesigner import *
import xarray as xr
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
model = MisInstrumentModel(SYSTEM, grat, gamma_ofst=GAMMA, alpha=-ALPHA)
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
# qe = np.loadtxt('imx533_rel_qe.csv', delimiter=',').T
# qe[0] *= 10  # nm -> Angstrom
# qe[1] *= 0.91  # max QE 91%
camera = MisCamera(np.pi*9**2,
                   1,
                   1/SCALE,
                   1,
                #    qe_curve=(qe[0], qe[1]),
                   )
model.set_camera(camera)
mmap = model.mosaic_map(unique=True)

# data = fits.open(watch('1650'))[1].data
# img = Image.fromarray(data)
# img.save('hmsa_origin_sky.png')
# img = Image.open('hmsa_origin_night.png')
img = Image.open('hmsa_origin_sky_fixed.png')
img = img.convert('F')
img = img.rotate(-.411, resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
# img = img.transpose(Image.FLIP_TOP_BOTTOM)
(width, height) = mmap.wavelength.shape[::-1]
fullimg = Image.new('F', mmap.wavelength.shape[::-1], color=np.nan)
allimg = Image.new('F', mmap.wavelength.shape[::-1], color=np.nan)
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
img = img.convert('F')
img = img.rotate(-.411, resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
# img = img.transpose(Image.FLIP_TOP_BOTTOM)
SCALE = 65.25
fullimg = Image.new('F', (width, height), color=np.nan)
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
# model.store('hmsa_origin_ship.json', True)
# print('Saved instrument model.')
# print('Finished.\n')
# %%
ret = MisCurveRemover(model)
# %%
img = xr.DataArray(np.asarray(allimg).astype(float), 
                   dims=['gamma', 'beta'],
                   coords={'gamma': ret.gamma_grid, 'beta': ret.beta_grid},
                   attrs={'unit': 'ADU'})
# %%
wls = {
    'Hβ': 486.1,
    'Hα': 656.3,
    '4278': 427.8,
    '5577': 557.7,
    '6300': 630.0,
    '7774': 777.4,
}
for window in ret.windows:
    data = MisCurveRemover.straighten_image(ret, img, window)
    fig, ax = plt.subplots()
    data.plot(ax=ax)
    ax.axvline(wls[window], color='red')
    plt.show()
    plt.close(fig)
# %%
