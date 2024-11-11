# %%
from __future__ import annotations
from matplotlib import pyplot as plt
import numpy as np
import tosholi
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
# img.plot_lines([
#     MisFeatures(6300, plot_styles={'color': 'red'}),
#     MisFeatures(5577, plot_styles={'color': 'green'}),
#     MisFeatures(7774, plot_styles={'color': 'brown'}),
#     MisFeatures(4278, plot_styles={'color': 'violet'}),
#     MisFeatures(6563, plot_styles={'color': 'orange'}),
#     MisFeatures(4861, plot_styles={'color': 'cyan'}),
#     7821, 7841, 6522, 6568
# ])
# %%
sol = xr.load_dataset('solar_spectra_air.nc')
source_wl = sol['wavelength'].values*10 # nm -> Angstrom
source_int = sol['irradiance'].values/10 # W/m^2/nm -> W/m^2/Angstrom

# %%
length = max(grat.mosaic.width, grat.mosaic.height)
dx = length / 1024
camera = MisCamera(254.5, 1, dx, np.inf, 1)
ret = img.simulate(source_wl, source_int, camera, report=False)
# plt.show()
# %%
img.intensity_plot(ret[0], [
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
], fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
# %%
img.order_map(ret[1], [
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
], fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
# %%
fig, ax = img._plot_lines(False, img.alpha, [
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
], mode='Mosaic', default_style={'ls': '-', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}, labels=True, fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
intensity = ret[0]
im = ax.imshow(intensity.values, origin='lower', extent=[intensity.beta.values[0], intensity.beta.values[-1], intensity.gamma.values[0], intensity.gamma.values[-1]])
# im = ax.pcolormesh(intensity.beta.values[::-1], intensity.gamma.values, intensity.values[:, ::-1], shading='auto')
fig.subplots_adjust(bottom=0.7)
cax = fig.add_axes([0.1, 0, 0.8, 0.01])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Intensity (e$^-$)')
cbar.formatter.set_useMathText(True)
# %%
(intensities, extra_maps), _, _, _ = ret
# %%
for k, v in extra_maps.items():
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
    im = ax.pcolormesh(v.beta.values, v.gamma.values, v['order'].values, shading='auto')
    fig.colorbar(im, ax=ax)
    ax.set_title(k)
    plt.show()
# %%
# %%
for k, v in extra_maps.items():
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
    vals = v['wavelength'].values
    vals[np.where(vals == 0)] = np.nan
    im = ax.pcolormesh(v.beta.values, v.gamma.values, vals, shading='auto')
    fig.colorbar(im, ax=ax)
    ax.set_title(k)
    plt.show()
# %%
output = []

for k in ret[1].items():
    k, v = k
    orders = np.unique(v['order'].values)
    for order in orders:
        if np.isnan(order):
            continue
        vsel = v.order.values.copy()
        beta = v.beta.values
        gamma = v.gamma.values
        bb, gg = np.meshgrid(beta, gamma)
        shape = vsel.shape
        vsel = vsel.flatten()
        bb = bb.flatten()
        gg = gg.flatten()
        idx = np.where(vsel != order)
        lidx = np.where(vsel == order)
        bb[idx] = np.nan
        gg[idx] = np.nan
        bb = bb.reshape(shape)
        gg = gg.reshape(shape)
        vsel = vsel.reshape(shape)
        output.append((k, order, bb, gg, len(lidx[0])))
        # ax.plot(bb, gg, color=colors[kidx], alpha=0.5)
        # ax.scatter(0, 0, color=colors[kidx], label=f'{k}:{order:.0f}')
        # kidx += 1
        # bmin = np.nanmin(bb, axis=1)
        # bmax = np.nanmax(bb, axis=1)
        # gamma = np.nanmean(gg, axis=1)
        # pts = np.concatenate((np.vstack((bmin, gamma)).T, np.vstack((bmax[::-1], gamma[::-1])).T))
        # print(pts)
        # patch = Polygon(pts, closed=True, fill=True, alpha=0.5)
        # ax.add_patch(patch)
        # ycoord = np.nanmean(gamma)
        # gidx = np.argmin(np.abs(gamma - ycoord))
        # xcoord = np.nanmean([bmin[gidx], bmax[gidx]])
        # ax.text(xcoord, ycoord, f'{k}:{order:.0f}', ha='center', va='center', color=colors[kidx])

def sortby(x):
    return x[-1]

output.sort(key=sortby)
output.reverse()
# %%
fig, ax = img._plot_lines(False, img.alpha, [
    MisFeatures(6300, plot_styles={'color': 'red'}),
    MisFeatures(5577, plot_styles={'color': 'green'}),
    MisFeatures(7774, plot_styles={'color': 'brown'}),
    MisFeatures(4278, plot_styles={'color': 'violet'}),
    MisFeatures(6563, plot_styles={'color': 'orange'}),
    MisFeatures(4861, plot_styles={'color': 'cyan'}),
    7821, 7841, 6522, 6568
], mode='Mosaic', default_style={'ls': '-', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}, labels=False, fig_kwargs={'figsize': (6.4, 5.6), 'dpi': 300})
cmap = plt.cm.gist_rainbow
norm = mpl.colors.Normalize(vmin=0, vmax=len(output) - 1)
colors = [cmap(norm(i)) for i in range(len(output))]
import random
random.shuffle(colors)
for kidx, op in enumerate(output):
    k, order, bb, gg, count = op
    p = ax.axhline(0, color=colors[kidx], label=f'{k}:{order:.0f}', zorder=0)
    ax.plot(bb, gg, color=p.get_color(), alpha=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
# %%
