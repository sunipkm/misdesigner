# %%
from __future__ import annotations
import tosholi
import matplotlib as mpl
from misdesigner import *
import yaml
from dataclasses import asdict

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
slit_height = 64.44
grat = MisGrating(400, 442.7, 98.76,
                  {
                      'BL': MisSlit(29, -slit_height / 4, 0.05, slit_height / 2), #, ranges=[(2800, 4600), (7100, 11000)]),
                      'BR': MisSlit(0, -slit_height / 4, 0.05, slit_height / 2), #, ranges=[(4950, 7000)]),
                      'TL': MisSlit(29, slit_height / 4, 0.05, slit_height / 2), #, ranges=[(5900, np.inf)]),
                      'TR': MisSlit(0, slit_height / 4, 0.05, slit_height / 2) #, ranges=[(-np.inf, 5500)]),
                  },
                  MisMosaic((-5.85 - 58.35)*0.5, (-1.8+2.5)/2, 24.63 + 27.57 + 0.3,  27.72 + 26.96 + 1.8 + 2.5,
                            [
                                MisMosaicFilter(0, 1.8, 24.63, 26.96, [(4861-100, 4861+100)], name='Hβ'), # Hβ, 20nm around 4861
                                MisMosaicFilter(24.63, 1.8, 27.57 - 2.68, 26.96, [(6563-100, 6563+100)], name = 'Hα'), # Hα, 20nm around 6563
                                MisMosaicFilter(0, 26.96 + 1.8, 8.13, 27.72, [(4300-50, 4300+50)], name='4278'), # OI, 10nm around 7774
                                MisMosaicFilter(8.13, 26.96 + 1.8, 8, 27.72, [(5580-50, 5580+50)], name='5577'), # OI, 10nm around 5580
                                MisMosaicFilter(8.13 + 8, 26.96 + 1.8, 7.5, 27.72, [(6300-50, 6300+50)], name='6300'), # OI, 10nm around 6300
                                MisMosaicFilter(8.13 + 8 + 7.5, 26.96 + 1.8, 28.57 - 2.68, 27.72, [(7774-50, 7774+50)], name = '7774'), # N2+, 10nm around 4300
                            ]))
img = LinePredictor('HMS-A ORIGIN', grat, gamma_ofst=0, alpha=-70.8)
# %%
img.plot_lines([
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
])
# %%
