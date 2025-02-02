# %%
from __future__ import annotations
from glob import glob
import os
import sys
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
import matplotlib as mpl
from misdesigner import *
import xarray as xr
from PIL import Image
from astropy.io import fits

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
SYSTEM = 'HMS-A ECLIPSE'
SLIT_WIDTH = 250
SLIT_TO_EDGE = -13.83 # -5.93  # -3.55 -> -70.8 (α)
EDGE_TO_MOSAIC = -2.3
MOSAIC_WIDTH = 52.5
MOSAIC_HEIGHT = 55.19
MOSAIC_X0 = SLIT_TO_EDGE + EDGE_TO_MOSAIC
MOSAIC_X1 = MOSAIC_X0 - MOSAIC_WIDTH
slit_height = 80
FL1 = 435
FL2 = 435
ALPHA = 67.55
# FL1 = 420
# FL2 = 430
# ALPHA = 71.08
GAMMA = 0
grat = MisConfig(FL1, FL2, 98.76,
                 {
                     # (7150, 10950)]),
                     'BL': MisSlit(21.43, -slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(3150, 4450), (7000, 8000)]),
                     'BR': MisSlit(0, -slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(4950, 10000)]),
                     'TL': MisSlit(21.43, slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(5900, 10000)]),
                     'TR': MisSlit(0, slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(4350, 5000)]),
                 },
                 MisMosaic((MOSAIC_X0 + MOSAIC_X1)*0.5, (-1.94+2.68),
                           MOSAIC_WIDTH,  MOSAIC_HEIGHT,
                           [
                     # Hβ, 20nm around 4861
                     MisMosaicFilter(0, 1.94, 29.26, 25.02, [
                         (4860-50, 4861+50)], name='Hβ'),
                     # Hα, 20nm around 6563
                     MisMosaicFilter(
                         29.26, 1.94, 20.24, 25.02, [(6500-125, 6563+125)], name='Hα'),
                     # O I, 10 nm around 7800, 252
                     MisMosaicFilter(
                         0, 1.94 + 25.02, 13.9, 25.02, [(7774-50, 7774+50)], name='7774'
                    ),
                    # O I, 10 nm around 5600, 415
                    MisMosaicFilter(
                        13.9, 1.94 + 25.02, 8.91, 25.02, [(5600-50, 5600+50)], name='5577'
                    ),
                    # O I, 10 nm around 6320, 583
                    MisMosaicFilter(
                        13.9 + 8.91, 1.94 + 25.02, 9.2, 25.02, [(6320-50, 6320+50)], name='6300'
                    ),
                    # N2+, 10 nm around 4300, 905
                     MisMosaicFilter(
                         13.9 + 8.91 + 9.2, 1.94 + 25.02, 17.5, 25.02, [(4300-50, 4300+50)], name='4278'),
                 ]))
model = MisInstrumentModel(SYSTEM, grat, gamma_ofst=GAMMA, alpha=-ALPHA)

fig, ax = model.plot_lines([
    MisFeatures(6300, plot_styles={'color': 'red', 'lw': 1}),
    MisFeatures(5577, plot_styles={'color': 'green', 'lw': 1}),
    MisFeatures(7774, plot_styles={'color': 'brown', 'lw': 1}),
    MisFeatures(4278, plot_styles={'color': 'violet', 'lw': 1}),
    MisFeatures(6563, plot_styles={'color': 'orange', 'lw': 1}),
    MisFeatures(4861, plot_styles={'color': 'cyan', 'lw': 1}),
],
    fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300}, 
    modify=True)

SCALE = 400/20.24
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
# data = fits.open('hitmis_14ms_0_0_1715118908415.fit')[1].data
# data = fits.open('atik_124ms_1643139689842.fit')[1].data
# data = fits.open('atik_120000ms_1643150338697.fit')[1].data
data = fits.open('hitmis_30000ms_0_0_1712604563551.fit')[1].data
# img = Image.fromarray(data)
# img.save('hmsa_origin_sky.png')
# img = Image.open('hmsa_origin_night.png')
img = Image.fromarray(data)
img.save('hmsa_eclipse_ref.png')
img = img.convert('F')
# img = img.rotate(0.5, resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
# img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
(width, height) = mmap.wavelength.shape[::-1]
fullimg = Image.new('F', mmap.wavelength.shape[::-1], color=np.nan)
nwidth = width / SCALE
nheight = height / SCALE
fullimg.paste(img, (45, -65))
# allimg.paste(fullimg.crop((0, 0, width, height/2)), (0, 0))
# fullimg = np.asarray(fullimg)
# vmin = np.percentile(fullimg, 1)
# vmax = np.percentile(fullimg, 99.5)
# ax.imshow(np.asarray(fullimg)[0:height//2],
#           origin='lower', extent=(0, -nwidth, 0, nheight/2),
#           aspect='equal',
#           #   vmin=vmin, vmax=vmax,
#           zorder=0)
# img = Image.open('hmsa_origin_night.png')
# img = img.convert('F')
# img = img.rotate(-.411, resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
# img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
# # img = img.transpose(Image.FLIP_TOP_BOTTOM)
# SCALE = 65.25
# fullimg = Image.new('F', (width, height), color=np.nan)
# fullimg.paste(img, (110, 410))
# allimg.paste(fullimg.crop((0, height/2, width, height)), (0, height//2))
# # fullimg = np.asarray(fullimg)
# # vmin = np.percentile(fullimg, 1)
# # vmax = np.percentile(fullimg, 99.5)
# # ax.imshow(np.asarray(fullimg)[height//2:],
# #           origin='lower', extent=(0, -nwidth, nheight/2, nheight),
# #           aspect='equal',
# #           #   vmin=vmin, vmax=vmax,
# #           zorder=0)
iarray = np.asarray(fullimg)
vmin = np.nanpercentile(iarray, 1)
vmax = np.nanpercentile(iarray, 99)
print(vmin, vmax)
im = ax.imshow(iarray,
          origin='lower', extent=(0, -nwidth, 0, nheight),
          aspect='equal',
          vmin=vmin, vmax=vmax,
          zorder=0,
          cmap='bone')
fig.colorbar(im)
# plt.savefig('hmsa_origin_ref.png', dpi=2400, bbox_inches='tight', transparent=True)
plt.show()
# %%
model.store('hmsa_eclipse.json', True)
# %%
