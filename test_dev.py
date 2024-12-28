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
model = InstrumentModel.load('hmsa_origin.json')
for k, v in model.slits.items():
    SLIT_WIDTH = round(v.width*1e3)
    break
SYSTEM = model.hmsVersion
# %%
sol = xr.load_dataset('solar_spectra_air.nc')
source_wl = sol.wavelength.values*10 # nm -> Angstrom
source_int = sol.irradiance.values/10 # W/m^2/nm -> erg/s/cm^2/Angstrom
# %%
print('Starting image simulation...', end=' ')
sys.stdout.flush()
start = perf_counter_ns()
ret = model.simulate(source_wl, source_int, report=True)
shape = ret.total_intensity.shape
print('Done.')
print(f"Time to simulate ({shape[0]} x {shape[1]}): {(perf_counter_ns() - start) / 1e9} s")
# %%
fig, _, _ = model.intensity_plot(ret.total_intensity, [
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
], fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
fig.savefig(f'{SYSTEM}_intensity_{SLIT_WIDTH}.png', dpi=300, bbox_inches='tight')
print('Saved intensity plot.')
plt.show()
# %%
fig, _ = model.intensity_plot_rgb(ret, [
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
], fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
fig.savefig(f'{SYSTEM}_intensity_rgb_{SLIT_WIDTH}.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved RGB intensity plot.')
# %%
fig, _ = model.order_map(ret, [
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
], fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
fig.savefig(f'{SYSTEM}_order_map_all_{SLIT_WIDTH}.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved total order map.')
# %%
for slit in ret.slit.values:
    model.order_map_slit(ret, slit, fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
    plt.savefig(f'{SYSTEM}_order_map_{slit}_{SLIT_WIDTH}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved order map for slit {slit}.')
# %%
model.store('hmsa_origin.json', True)
print('Saved instrument model.')
print('Finished.\n')
# %%
