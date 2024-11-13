# %%
from __future__ import annotations
import sys
from time import perf_counter_ns
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from misdesigner import *
import yaml
from dataclasses import asdict
import xarray as xr

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
SYSTEM = 'HMS-A ORIGIN'
SLIT_WIDTH = 75
SLIT_TO_EDGE = -7 # -3.55 -> -70.8 (α)
EDGE_TO_MOSAIC = -2.3
MOSAIC_WIDTH = 52.5
MOSAIC_X0 = SLIT_TO_EDGE + EDGE_TO_MOSAIC
MOSAIC_X1 = MOSAIC_X0 - MOSAIC_WIDTH
slit_height = 66.5
grat = MisConfig(400, 442.7, 98.76,
                  {
                      'BL': MisSlit(29, -slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(3150, 4450)]), # (7150, 10950)]),
                      'BR': MisSlit(0, -slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(4950, np.inf)]),
                      'TL': MisSlit(29, slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(5900, np.inf)]),
                      'TR': MisSlit(0, slit_height / 4, SLIT_WIDTH*1e-3, slit_height / 2, ranges=[(4350, 5000)]),
                  },
                  MisMosaic((MOSAIC_X0 + MOSAIC_X1)*0.5, (-1.8+2.5), 24.63 + 27.57 + 0.3,  27.72 + 26.96 + 1.8 + 2.5,
                            [
                                MisMosaicFilter(0, 1.8, 24.63, 26.96, [(4860-50, 4861+50)], name='Hβ'), # Hβ, 20nm around 4861
                                MisMosaicFilter(24.63, 1.8, 27.57 - 2.68, 26.96, [(6500-125, 6563+125)], name = 'Hα'), # Hα, 20nm around 6563
                                MisMosaicFilter(0, 26.96 + 1.8, 8.13, 27.72, [(4300-50, 4300+50)], name='4278'), # OI, 10nm around 7774
                                MisMosaicFilter(8.13, 26.96 + 1.8, 8, 27.72, [(5500-125, 5500+125)], name='5577'), # OI, 10nm around 5580
                                MisMosaicFilter(8.13 + 8, 26.96 + 1.8, 7.5, 27.72, [(6320-50, 6320+50)], name='6300'), # OI, 10nm around 6300
                                MisMosaicFilter(8.13 + 8 + 7.5, 26.96 + 1.8, 28.57 - 2.68, 27.72, [(7750-125, 7750+125)], name = '7774'), # N2+, 10nm around 4300
                            ]))
model = InstrumentModel(SYSTEM, grat, gamma_ofst=0, alpha=-71.1)
# %%
model.scan_lines([
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
])
# %%
sol = xr.load_dataset('solar_spectra_air.nc')
source_wl = sol['wavelength'].values*10 # nm -> Angstrom
source_int = sol['irradiance'].values/10 # W/m^2/nm -> W/m^2/Angstrom

# %%
length = max(grat.mosaic.width, grat.mosaic.height)
IMG_SZ = 1024
dx = length / IMG_SZ
camera = MisCamera(254.5, 1, dx, 1)
# %%
model.intensity_model(source_wl, source_int, camera)
# %%
# %%
model.store('hmsa_origin.json', True)
print('Saved instrument model.')
print('Finished.\n')
# %%
