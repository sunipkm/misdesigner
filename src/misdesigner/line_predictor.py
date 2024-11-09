# %%
from __future__ import annotations
from typing import Dict, List, Literal, Optional, SupportsFloat as Numeric, Tuple
import os
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
import tosholi
import astropy.io.fits as pf
from scipy.optimize import curve_fit
from skimage.exposure import equalize_hist
import xarray as xr
import matplotlib as mpl
import matplotlib.widgets as mpl_widgets

from .instrument_params import MisCamera, MisGrating, MisFeatures, MisMosaic, MisMosaicFilter, MisSlit, MisGratingCfg, MisInstrument
# %%

PlotMode = Literal['Angle', 'Mosaic']


class LinePredictor(MisGrating):
    @staticmethod
    def load(configfile: str, alpha: Optional[Numeric] = None, *, gamma_ofst: Optional[Numeric] = None) -> LinePredictor:
        if not os.path.exists(configfile):
            raise FileNotFoundError(f"File {configfile} not found.")
        ext = os.path.splitext(configfile)[-1].lower()
        if ext == '.toml':
            with open(configfile, 'rb') as ifile:
                params = tosholi.load(MisInstrument, ifile)
        else:
            raise TypeError(
                f"Invalid file extension {ext}. Please provide a .toml file.")
        return LinePredictor(params.system, params.optics, params.instrument.alpha, gamma_ofst=params.instrument.gamma_ofst, input_wls=params.lines)

    def store(self, path: str, overwrite: bool = False):
        instr = MisGratingCfg(self.alpha, self.gamma_ofst)
        params = MisInstrument(self.hmsVersion, self, instr, self.input_wls)
        dirname = os.path.dirname(path)
        if len(dirname) > 0 and not os.path.exists(dirname):
            os.makedirs(dirname)
        if not overwrite and os.path.exists(path) and os.path.isfile(path):
            raise FileExistsError(f"File {path} already exists.")
        with open(path, 'wb') as ofile:
            tosholi.dump(params, ofile)

    def get_instrument_params(self) -> MisInstrument:
        instr = MisGratingCfg(self.alpha, self.gamma_ofst)
        return MisInstrument(self.hmsVersion, self, instr, self.input_wls)

    def __init__(
            self, system: str,
            optics: MisGrating,
            alpha: Optional[Numeric] = None,
            *,
            gamma_ofst: Numeric = 0,
            alpha_min: Numeric = -90,
            alpha_max: Numeric = 0,
            alpha_step: Numeric = 0.1,
            n_beta: int = 100,
            n_gamma: int = 100,
            input_wls: Optional[List[MisFeatures]] = []):
        super().__init__(
            optics.fl_collimator,
            optics.fl_mosaic,
            optics.sigma,
            optics.slits,
            optics.mosaic
        )

        if alpha is not None:
            self.alpha = alpha
        self.gamma_ofst = gamma_ofst
        self.input_wls = input_wls

        if alpha_min > alpha_max:
            alpha_min, alpha_max = alpha_max, alpha_min
        self.grating_angle_min = alpha_min
        self.grating_angle_max = alpha_max
        self.grating_angle_step = abs(alpha_step)

        self.n_beta = n_beta
        self.n_gamma = n_gamma

        self.den = 1e7/self.sigma  # groove distance in Angstrom

        self.hmsVersion = system.upper()
        self.gammas = self.relative_gammas(
            self.gamma_ofst)  # grating coordinate
        dbeta = np.rad2deg(np.arctan(self.mosaic.width/2/self.fl_mosaic))
        betaofst = np.rad2deg(np.arctan(self.mosaic.x/self.fl_mosaic))
        self.beta_min = betaofst - dbeta
        self.beta_max = betaofst + dbeta
        if self.beta_min > self.beta_max:
            self.beta_min, self.beta_max = self.beta_max, self.beta_min
        dgamma = np.rad2deg(np.arctan(self.mosaic.height/2/self.fl_mosaic))
        mgamma_ofst = np.rad2deg(np.arctan(self.mosaic.y/self.fl_mosaic))
        self.gamma_min = mgamma_ofst - dgamma
        self.gamma_max = mgamma_ofst + dgamma
        if self.gamma_min > self.gamma_max:
            self.gamma_min, self.gamma_max = self.gamma_max, self.gamma_min
        self.mosaic_bottom_left = (self.image_deg_to_mm(
            self.beta_max), self.image_deg_to_mm(self.gamma_min))  # in instrument coord
        self.blurs = self.slit_blurs()

    def image_deg_to_mm(self, deg):
        return np.tan(np.deg2rad(deg))*self.fl_mosaic

    def relative_alphas(self, alpha):
        alphas = {}
        for k, v in self.slits.items():
            alphas[k] = alpha + \
                np.rad2deg(np.arctan(v.x / self.fl_collimator))
        return alphas

    def slit_blurs(self):
        return {k: 2*np.arctan(v.width/2/self.fl_collimator) for k, v in self.slits.items()}

    def relative_alpha(self, slit: str, grating_angle: Numeric | np.ndarray) -> Numeric | np.ndarray:
        v = self.slits[slit]
        ofst = np.rad2deg(np.arctan(v.x / self.fl_collimator))
        return ofst - grating_angle

    def relative_gammas(self, gamma):  # grating coordinate
        gammas = {}
        for k, v in self.slits.items():
            min_gamma = 90 + \
                (-gamma +
                 np.rad2deg(np.arctan((v.y - v.height/2) / self.fl_collimator)))
            max_gamma = 90 + \
                (-gamma +
                 np.rad2deg(np.arctan((v.y + v.height/2) / self.fl_collimator)))
            if min_gamma < max_gamma:
                min_gamma, max_gamma = max_gamma, min_gamma
            gammas[k] = (min_gamma, max_gamma)
        return gammas

    def min_gamma(self):
        return min([v[0] for v in self.gammas.values()])

    def max_gamma(self):
        return max([v[1] for v in self.gammas.values()])

    def grating_eqn(self, alpha, gamma, m, wl):
        """## Grating Equation

        ### Args:
            - `alpha (Numeric)`: Incident angle (degrees).
            - `gamma (Numeric | np.ndarray)`: Incident angle parallel to the grating (degrees).
            - `m (int)`: Diffraction order.
            - `wl (Numeric)`: Wavelength in Angstrom.

        ### Returns:
            - `Numeric | np.ndarray`: Diffracted angle β.
        """
        const = m*wl/(self.den*np.sin(np.deg2rad(gamma)))
        sinb = const - np.sin(np.deg2rad(alpha))
        sinb[np.where((sinb > 1) | (sinb < -1))] = np.nan
        beta = np.arcsin(sinb)
        return np.rad2deg(beta)

    def gamma_to_image(self, gamma):
        # for a reflective grating
        # additional -90 from going back to instrument coordinate
        val = 90 - gamma
        vlen = np.tan(np.deg2rad(val))*self.fl_collimator
        return np.rad2deg(np.arctan(vlen / self.fl_mosaic))

    def beta_to_image(self, beta, grating_angle):
        return beta + grating_angle

    def beta_to_grating(self, beta, grating_angle):
        return beta - grating_angle

    def grating_product(self, slit: str, grating_angle: Numeric, *, n_beta: int = 100, n_gamma: int = 100) -> xr.Dataset:
        alpha = self.relative_alpha(slit, grating_angle)  # grating coord
        if alpha < -90 or alpha > 90:
            warnings.warn(
                f'Slit {slit} is not illuminated at grating angle {grating_angle} deg.')
        gmin, gmax = self.gammas[slit]
        minb, maxb = self.beta_min, self.beta_max
        gamma = np.linspace(gmin, gmax, int(n_gamma))  # grating coordinate
        beta = np.linspace(minb, maxb, int(n_beta))  # instrument coordinate
        beta_ = self.beta_to_grating(beta, grating_angle)  # grating coordinate
        d_beta = np.mean(np.diff(beta))
        if d_beta > 4:
            warnings.warn(
                'Large beta step size. Consider reducing the step size for better accuracy.')
        bb, gg = np.meshgrid(beta_, gamma)
        alph = np.sin(np.deg2rad(alpha))
        gam = np.sin(np.deg2rad(gg))
        bet = np.sin(np.deg2rad(bb))
        prod = (bet + alph) * gam * self.den
        dnx = self.den * gam * np.cos(np.deg2rad(bb)) * np.deg2rad(d_beta)
        return xr.Dataset(
            {
                'grating_product': (['beta', 'gamma'], prod),
                'd_nx': (['beta', 'gamma'], dnx),
                'beta_arr': (['beta', 'gamma'], bb),
                'gamma_arr': (['beta', 'gamma'], gg),
            },
            coords={'gamma': self.gamma_to_image(gamma), 'beta': beta},
            attrs={
                'slit': slit,
                'alpha': alpha,
                'gamma': self.gammas[slit]
            }
        )

    def get_lines(self, prod: xr.Dataset, grating_angle: Numeric, wavelength: int, blur: Numeric):
        n_frac, n_int = np.modf(prod.grating_product.values / wavelength)
        n_int = n_int.astype(int)
        n_frac /= prod.d_nx.values / wavelength
        n_frac = np.abs(n_frac)
        ords = np.unique(n_int)
        valid_ords = []
        for ord in ords:
            idx = np.where((n_int == ord) & (n_frac < 1))
            if len(idx[0]) > 0:
                valid_ords.append(ord)
        lines: List[Tuple[int, np.ndarray, np.ndarray]] = []
        # calculate lines using grating eqn
        for ord in valid_ords:
            beta = self.grating_eqn(
                prod.attrs['alpha'],
                prod.gamma_arr.values[:, 0],
                ord, wavelength)  # grating coordinate
            res = ord*wavelength/(self.den*np.cos(np.deg2rad(beta))
                                  * self.blurs[prod.attrs['slit']])  # mλ/(d*cos(β))
            # res = np.sin(np.deg2rad(prod.gamma_arr.values[:, 0])) \
            #     * (np.sin(np.deg2rad(beta)) + np.sin(np.deg2rad(prod.attrs['alpha']))) \
            #     / (self.blurs[prod.attrs['slit']]*np.cos(np.deg2rad(beta))) # sin(γ)*(sin(β)+sin(α))/(B*cos(β))
            # instrument coordinate?
            beta = self.beta_to_image(beta, grating_angle)
            lines.append((ord, beta, prod.gamma.values, res))
        return lines

    def plot_lines(self, wavelengths: List[int | MisFeatures] = None, *, mode: PlotMode = 'Mosaic', default_style={'ls': '-.', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}, alpha: Optional[Numeric] = None, **fig_kwargs):
        NUM_COLORS = 10
        cmap = plt.cm.gist_rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
        colors = [cmap(norm(i)) for i in range(NUM_COLORS)]

        if alpha is not None:
            self.alpha = alpha
        elif not hasattr(self, 'alpha'):
            self.alpha = 0

        self.alpha_orig = self.alpha

        if wavelengths is not None:
            self.input_wls = wavelengths.copy()
            for k, v in enumerate(wavelengths):
                if isinstance(v, int):
                    self.input_wls[k] = MisFeatures(v)
            wls = wavelengths
        else:
            wls = self.input_wls.copy()
        for k, v in enumerate(wls):
            if isinstance(v, int):
                style = default_style.copy()
                if len(colors) > 0:
                    style['color'] = colors.pop()
                wls[k] = MisFeatures(v, plot_styles=style)
            elif isinstance(v, MisFeatures):
                if v.plot_styles is None:
                    style = default_style.copy()
                    if len(colors) > 0:
                        style['color'] = colors.pop()
                    v.plot_styles = style
                else:
                    if v.plot_styles['color'] is None:
                        if len(colors) > 0:
                            v.plot_styles['color'] = colors.pop()
                    if 'ls' not in v.plot_styles:
                        v.plot_styles['ls'] = default_style['ls']
                    if 'lw' not in v.plot_styles:
                        v.plot_styles['lw'] = default_style['lw']
                    if 'ms' not in v.plot_styles:
                        v.plot_styles['ms'] = default_style['ms']
            else:
                raise TypeError(
                    f"Invalid type for index {k}: {v}. Expected int or HmsWlParam.")

        if 'constrained_layout' in fig_kwargs:
            fig_kwargs.pop('constrained_layout')
        fig = plt.figure(constrained_layout=True, **fig_kwargs)
        gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.1, height_ratios=[
                              0.6, 0.05, 0.05], width_ratios=[1, 0.25, 0.25])
        ax = fig.add_subplot(gs[0, :])

        axalpha = fig.add_subplot(gs[1, :])
        alpha_slider = mpl_widgets.Slider(
            ax=axalpha,
            label=r'$\alpha$ ($^\circ$)',
            valmin=self.grating_angle_min,
            valmax=self.grating_angle_max,
            valstep=self.grating_angle_step,
            valinit=self.alpha,
            valfmt='%+05.1f',
        )

        resetax = fig.add_subplot(gs[2, -1])
        button = mpl_widgets.Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            alpha_slider.reset()

        button.on_clicked(reset)

        self._wls = wls

        self._update_alpha_plot_lines(self.alpha, fig, ax, mode)

        alpha_slider.on_changed(lambda x: self._update_alpha_plot_lines(x, fig, ax, mode))

        if mode == 'Angle':
            fig.suptitle(f'{self.hmsVersion}\ANGLE')
            ax.set_ylim(self.gamma_min, self.gamma_max)
            ax.set_xlim(self.beta_min, self.beta_max)
            ax.set_xlabel(r'$\beta$ ($^\circ$)', fontdict={'size': 10})
            ax.set_ylabel(r'$\gamma$ ($^\circ$)', fontdict={'size': 10})
        elif mode == 'Mosaic':
            fig.suptitle(f'{self.hmsVersion}\nMOSAIC')
            ax.set_xlim(-self.mosaic.width, 0)
            ax.set_ylim(0, self.mosaic.height)
            ax.invert_xaxis()
            ax.set_xlabel(r'$\beta$ (mm)', fontdict={'size': 10})
            ax.set_ylabel(r'$\gamma$ (mm)', fontdict={'size': 10})
            if self.mosaic.windows is not None:
                for window in self.mosaic.windows:
                    ax.add_patch(plt.Rectangle(
                        (-window.x-window.width, window.y), window.width, window.height, edgecolor='black', facecolor='none', lw=0.5, zorder=0))
                    if window.name is not None:
                        ax.text(-window.x, window.y+window.height, window.name,
                                va='top', ha='left', zorder=0, color='black')
        ax.set_aspect('equal')
        plt.show()

    def _update_alpha_plot_lines(self, alpha, fig: plt.Figure, ax: plt.Axes, mode: PlotMode):
        self.alpha = alpha

        if hasattr(self, '_lines'):
            del self._lines

        if hasattr(self, '_annot'):
            self._annot.remove()

        if hasattr(self, '_layout'):
            del self._layout

        for line in ax.get_lines():
            line.remove()

        lines = []

        annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points", fontsize=8,
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"), zorder=100)
        annot.set_visible(False)

        def update_annot(line, ind, slit, ord, wl, res):
            x, y = line.get_data()
            idx = ind["ind"][0]
            annot.xy = (x[idx], y[idx])
            text = "{}".format(
                ",".join([slit, str(wl), str(ord), f'{abs(res[idx]*1e-3):.0f}k']))
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                for v in self._lines:
                    slit, line, ord, wl, res = v
                    cont, ind = line.contains(event)
                    if cont:
                        update_annot(line, ind, slit, ord, wl, res)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        break
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

        for slit in self.slits.keys():
            prod = self.grating_product(
                slit, self.alpha, n_beta=self.n_beta, n_gamma=self.n_gamma)
            for k, v in enumerate(self._wls):
                wl = v.wavelength
                visible = True
                # filter by feature from slit
                if v.slit_key is not None and len(v.slit_key) > 0:
                    if v.slit_key != slit:
                        continue
                else:  # filter by slit wavelength range
                    if self.slits[slit].ranges is not None and len(self.slits[slit].ranges) > 0:
                        visible = False
                        for r in self.slits[slit].ranges:
                            if wl >= r[0] and wl <= r[1]:
                                visible = True
                                break
                    if not visible:
                        continue
                # accepted
                to_plot = self.get_lines(
                    prod, self.alpha, wl, self.blurs[slit])
                for ord, beta, gamma, res in to_plot:
                    plotted = False
                    if mode == 'Mosaic':
                        beta = self.image_deg_to_mm(
                            beta) - self.mosaic_bottom_left[0]
                        gamma = self.image_deg_to_mm(
                            gamma) - self.mosaic_bottom_left[1]
                        if self.mosaic.windows is not None:
                            for window in self.mosaic.windows:
                                plotted = True
                                valid = window.check_position(wl, beta, gamma)
                                if not np.any(valid):
                                    continue
                                line, = ax.plot(
                                    beta[valid], gamma[valid], **v.plot_styles)
                                lines.append((slit, line, ord, wl, res))
                    if not plotted:
                        line, = ax.plot(beta, gamma, **v.plot_styles)
                        lines.append((slit, line, ord, wl, res))

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.draw_idle()
        self._lines = lines
        self._annot = annot

    def simulate(self, source: xr.Dataset, camera: MisCamera, wavelengths: List[int | MisFeatures] = None, *, default_style={}, alpha: Optional[Numeric] = None, **fig_kwargs):
        if alpha is not None:
            self.alpha = alpha
        # TODO: Set up plot

        # TODO: Set up sliders


        return
# %%
