# %%
from __future__ import annotations
from datetime import timedelta
import sys
from time import perf_counter_ns
from typing import Dict, List, Literal, Optional, SupportsFloat as Numeric, Tuple
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tosholi
from xarray import DataArray, Dataset
import matplotlib as mpl
import matplotlib.widgets as mpl_widgets

from .instrument_params import MisCamera, MisGrating, MisFeatures, MisMosaic, MisMosaicFilter, MisSlit, MisGratingCfg, MisInstrument
from .utils import common_range, sign_ceil, sign_floor

from .multi_integrate import multi_integrate, unsort
# %%

PlotMode = Literal['Angle', 'Mosaic']


class LinePredictor(MisGrating):
    """## The core of the MISDesigner package.
    This class is used to predict the spectral lines on the image plane for a given set of wavelengths.
    The class is initialized with the instrument optics parameters and the grating parameters.
    """
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
            n_beta: int = 200,
            n_gamma: int = 100,
            input_wls: Optional[List[MisFeatures]] = []):
        """## Initialize the LinePredictor object.

        ### Args:
            - `system (str)`: Name of the instrument.
            - `optics (MisGrating)`: Instrument optics parameters. Describes the position and size of the slits and the wavelength of light they let through, as well as the imaging mosaic position and size, and the individual mosaic filters if any. The grating density, the focal distance from the slits to the grating illumination collimator, as well as the focal distance of the post-grating lens to the mosaic are also included. The mosaic is at the focal plane of the post-grating lens.
            - `alpha (Optional[Numeric], optional)`: The grating rotation angle, along the axis parallel to the grooves, in degrees. Defaults to None.
            - `gamma_ofst (Numeric, optional)`: The tilt of the grating, perpendicular to the grooves, in degrees. Defaults to 0.
            - `alpha_min (Numeric, optional)`: Minimum allowed grating rotation angle, in degrees. Defaults to -90.
            - `alpha_max (Numeric, optional)`: Maximum allowed grating rotation angle, in degrees. Defaults to 0. These values constrain the diffraction orders in the positive range.
            - `alpha_step (Numeric, optional)`: Minimum step size of the grating rotation angle. Defaults to 0.1.
            - `n_beta (int, optional)`: Number of points along the dispersion axis. Defaults to 200.
            - `n_gamma (int, optional)`: Number of points along the translation axis. Defaults to 100.
            - `input_wls (Optional[List[MisFeatures]], optional)`: Wavelength features of interest. Defaults to [].
        """
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
        self.gammas = self._relative_gammas(
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
        self.mosaic_bottom_left = (self._image_deg_to_mm(
            self.beta_max), self._image_deg_to_mm(self.gamma_min))  # in instrument coord
        self.blurs = self._slit_blurs()

    def _image_deg_to_mm(self, deg):
        return np.tan(np.deg2rad(deg))*self.fl_mosaic

    def _image_mm_to_deg(self, mm):
        return np.rad2deg(np.arctan(mm / self.fl_mosaic))

    def _relative_alphas(self, alpha):
        alphas = {}
        for k, v in self.slits.items():
            alphas[k] = alpha + \
                np.rad2deg(np.arctan(v.x / self.fl_collimator))
        return alphas

    def _slit_blurs(self):
        return {k: 2*np.arctan(v.width/2/self.fl_collimator) for k, v in self.slits.items()}

    def _relative_alpha(self, slit: str, grating_angle: Numeric | np.ndarray) -> Numeric | np.ndarray:
        v = self.slits[slit]
        ofst = np.rad2deg(np.arctan(v.x / self.fl_collimator))
        return ofst - grating_angle

    def _relative_gammas(self, gamma):  # grating coordinate
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

    def _min_gamma(self):
        return min([v[0] for v in self.gammas.values()])

    def _max_gamma(self):
        return max([v[1] for v in self.gammas.values()])

    def _grating_eqn(self, alpha, gamma, m, wl):
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

    def _gamma_to_image(self, gamma):
        """Converts gamma in grating coordinate (degrees) to image coordinate (degrees)."""
        # for a reflective grating
        # additional -90 from going back to instrument coordinate
        val = 90 - gamma
        vlen = np.tan(np.deg2rad(val))*self.fl_collimator
        return np.rad2deg(np.arctan(vlen / self.fl_mosaic))

    def _gamma_to_mosaic(self, gamma):
        """Converts gamma in image coordinate (degrees) to mosaic coordinate (mm, relative to mosaic, origin at bottom-left corner)."""
        return self._image_deg_to_mm(gamma) - self.mosaic_bottom_left[1]

    def _gamma_from_mosaic(self, gamma):
        """Converts gamma in mosaic coordinate (mm, relative to mosaic, origin at bottom-left corner) to image coordinate (degrees)."""
        return self._image_mm_to_deg(gamma + self.mosaic_bottom_left[1])

    def _gamma_from_image(self, gamma):
        vlen = np.tan(np.deg2rad(gamma))*self.fl_mosaic
        val = np.rad2deg(np.arctan(vlen / self.fl_collimator))
        return 90 + val

    def _beta_from_mosaic(self, beta):
        """Converts beta in mosaic coordinate (mm, relative to mosaic, origin at bottom-left corner) to image coordinate (degrees)."""
        return self._image_mm_to_deg(beta + self.mosaic_bottom_left[0])

    def _beta_to_mosaic(self, beta):
        """Converts beta in image coordinate (degrees) to mosaic coordinate (mm, relative to mosaic, origin at bottom-left corner)."""
        return self._image_deg_to_mm(beta) - self.mosaic_bottom_left[0]

    def _beta_to_image(self, beta, grating_angle):
        return beta + grating_angle

    def _beta_to_grating(self, beta, grating_angle):
        return beta - grating_angle

    def _grating_product_lines(self, slit: str, grating_angle: Numeric, *, n_beta: int, n_gamma: int) -> Dataset:
        alpha = self._relative_alpha(slit, grating_angle)  # grating coord
        if alpha < -90 or alpha > 90:
            warnings.warn(
                f'Slit {slit} is not illuminated at grating angle {grating_angle} deg.')
            return None
        gmin, gmax = self.gammas[slit]
        minb, maxb = self.beta_min, self.beta_max
        gamma = np.linspace(gmin, gmax, int(n_gamma))  # grating coordinate
        beta = np.linspace(minb, maxb, int(n_beta))  # instrument coordinate
        beta_ = self._beta_to_grating(
            beta, grating_angle)  # grating coordinate
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
        return Dataset(
            {
                'grating_product': (['gamma', 'beta'], prod),
                'd_nx': (['gamma', 'beta'], dnx),
                'beta_arr': (['gamma', 'beta'], bb),
                'gamma_arr': (['gamma', 'beta'], gg),
            },
            coords={'gamma': self._gamma_to_image(gamma), 'beta': beta},
            attrs={
                'slit': slit,
                'alpha': alpha,
                'gamma': self.gammas[slit]
            }
        )

    def _grating_product_sim(self, slit: str, grating_angle: Numeric, beta_grid: np.ndarray, gamma_grid: np.ndarray) -> Dataset:
        alpha = self._relative_alpha(slit, grating_angle)  # grating coord
        if alpha < -90 or alpha > 90:
            warnings.warn(
                f'Slit {slit} is not illuminated at grating angle {grating_angle} deg.')
            return None
        # from mosaic coordinate to image coordinate
        gamma = self._gamma_from_mosaic(gamma_grid)
        # from image coordinate to grating coordinate
        gamma = self._gamma_from_image(gamma)
        # from mosaic coordinate to image coordinate
        beta = self._beta_from_mosaic(beta_grid)
        # from image coordinate to grating coordinate
        beta = self._beta_to_grating(beta, grating_angle)
        d_beta = np.mean(np.diff(beta))
        bb, gg = np.meshgrid(beta, gamma)
        gmin, gmax = self.gammas[slit]
        alph = np.sin(np.deg2rad(alpha))
        gg = np.sin(np.deg2rad(gg)) * self.den
        bb = np.deg2rad(bb)
        prod = (np.sin(bb) + alph) * gg
        dprod = gg * np.cos(bb) * np.deg2rad(d_beta)
        loc = np.where((gg > gmin) & (gg < gmax))
        prod[loc] = np.nan
        dprod[loc] = np.nan
        return Dataset(
            {
                'grating_product': (['gamma', 'beta'], prod),
                'd_nx': (['gamma', 'beta'], dprod),
                'intensity': (['gamma', 'beta'], np.zeros_like(prod)),
            },
            coords={
                'gamma': gamma_grid,
                'beta': beta_grid
            },
            attrs={
                'slit': slit,
                'alpha': alpha,
                'gmin': self._gamma_to_mosaic(self._gamma_to_image(self.gammas[slit][0])),
                'gmax': self._gamma_to_mosaic(self._gamma_to_image(self.gammas[slit][1])),
            }
        )

    def _get_lines(self, prod: Dataset, grating_angle: Numeric, wavelength: int, blur: Numeric):
        if prod is None:
            return []
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
            beta = self._grating_eqn(
                prod.attrs['alpha'],
                prod.gamma_arr.values[:, 0],
                ord, wavelength)  # grating coordinate
            res = ord*wavelength/(self.den*np.cos(np.deg2rad(beta))
                                  * self.blurs[prod.attrs['slit']])  # mλ/(d*cos(β))
            # res = np.sin(np.deg2rad(prod.gamma_arr.values[:, 0])) \
            #     * (np.sin(np.deg2rad(beta)) + np.sin(np.deg2rad(prod.attrs['alpha']))) \
            #     / (self.blurs[prod.attrs['slit']]*np.cos(np.deg2rad(beta))) # sin(γ)*(sin(β)+sin(α))/(B*cos(β))
            # instrument coordinate?
            beta = self._beta_to_image(beta, grating_angle)
            lines.append((ord, beta, prod.gamma.values, res))
        return lines

    def plot_lines(self, wavelengths: List[int | MisFeatures] = None, *, mode: PlotMode = 'Mosaic', default_style={'ls': '-.', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}, alpha: Optional[Numeric] = None, **fig_kwargs):
        """## Plot the given lines on the image plane, in angle or physical coordinates.
        This mode allows the user to visualize, and explore the line positions on the image plane
        for a given set of wavelengths by varying the grating angle.

        ### Args:
            - `wavelengths (List[int  |  MisFeatures], optional)`: Wavelength features of interest. Defaults to None. Must be provided if the object was not initialized with wavelengths (from a config file, or otherwise).
            - `mode (PlotMode, optional)`: Plot coordinates. Defaults to 'Mosaic'.
            - `default_style (dict, optional)`: Default style for plotting the spectral features. Defaults to dot-dashed lines, line width 0.5, marker size 0.2, color black. Note: The colors for the first 10 features without specified colors are automatically assigned.
            - `alpha (Optional[Numeric], optional)`: Initial grating angle in degrees. Must be a value between -90 deg and 90 deg. Defaults to None.
        """
        fig, ax = self._plot_lines(True, alpha, wavelengths, mode=mode,
                                   default_style=default_style, labels=False, labelcolor='black', fig_kwargs=fig_kwargs)
        return fig, ax

    def _plot_lines(self, sliders: bool, alpha: Numeric, wavelengths: List[int | MisFeatures], *, mode: PlotMode, default_style, labels, labelcolor, fig_kwargs) -> Tuple[plt.Figure, plt.Axes]:
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
                    if 'color' not in v.plot_styles:
                        v.plot_styles['color'] = default_style['color']
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

        if 'fig_kwargs' in fig_kwargs:
            fig_kwargs = fig_kwargs['fig_kwargs']
        if 'constrained_layout' in fig_kwargs:
            fig_kwargs.pop('constrained_layout')
        fig = plt.figure(constrained_layout=True, **fig_kwargs)
        if sliders:
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
        else:
            ax = fig.subplots()

        self._wls = wls

        self._update_alpha_plot_lines(self.alpha, fig, ax, mode, labels)

        if sliders:
            alpha_slider.on_changed(
                lambda x: self._update_alpha_plot_lines(x, fig, ax, mode, labels))

        if mode == 'Angle':
            fig.suptitle(f'{self.hmsVersion}\ANGLE')
            # instrument coordinate
            ax.set_ylim(self.gamma_min, self.gamma_max)
            ax.set_xlim(self.beta_min, self.beta_max)  # instrument coordinate
            ax.set_xlabel(r'$\beta$ ($^\circ$)', fontdict={'size': 10})
            ax.set_ylabel(r'$\gamma$ ($^\circ$)', fontdict={'size': 10})
        elif mode == 'Mosaic':
            if sliders:
                fig.suptitle(f'{self.hmsVersion}\nMOSAIC')
            else:
                fig.suptitle(
                    f'{self.hmsVersion}\nMOSAIC\n$\\alpha={self.alpha:.1f}^\\circ$')
            ax.set_xlim(-self.mosaic.width, 0)
            ax.set_ylim(0, self.mosaic.height)
            ax.invert_xaxis()
            ax.set_xlabel(r'$\beta$ (mm)', fontdict={'size': 10})
            ax.set_ylabel(r'$\gamma$ (mm)', fontdict={'size': 10})
            if self.mosaic.windows is not None:
                for window in self.mosaic.windows:
                    ax.add_patch(plt.Rectangle(
                        (-window.x-window.width, window.y), window.width, window.height, edgecolor='black', facecolor='none', lw=0.5, zorder=99))
                    if window.name is not None:
                        ax.text(-window.x, window.y+window.height, window.name,
                                va='top', ha='left', zorder=99, color=labelcolor)
        ax.set_aspect('equal')
        return (fig, ax)

    def _update_alpha_plot_lines(self, alpha, fig: plt.Figure, ax: plt.Axes, mode: PlotMode, labels: bool):
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
            prod = self._grating_product_lines(
                slit, self.alpha, n_beta=self.n_beta, n_gamma=self.n_gamma)
            for k, v in enumerate(self._wls):
                wl = v.wavelength
                visible = True
                if labels:
                    v.plot_styles['label'] = f'{v.wavelength:.0f}Å'
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
                to_plot = self._get_lines(
                    prod, self.alpha, wl, self.blurs[slit])
                for ord, beta, gamma, res in to_plot:
                    plotted = False
                    if mode == 'Mosaic':
                        beta = self._beta_to_mosaic(beta)
                        gamma = self._gamma_to_mosaic(gamma)
                        # beta = self.image_deg_to_mm(
                        #     beta) - self.mosaic_bottom_left[0]
                        # gamma = self.image_deg_to_mm(
                        #     gamma) - self.mosaic_bottom_left[1]
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
                        line, = ax.plot(
                            beta, gamma, **v.plot_styles, zorder=10)
                        lines.append((slit, line, ord, wl, res))

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.draw_idle()
        self._lines = lines
        self._annot = annot

    def simulate(self,
                 source_wl: np.ndarray, source_i: np.ndarray,
                 camera: MisCamera, *,
                 alpha: Optional[Numeric] = None,
                 report: bool = True,
                 use_c: bool = True) -> Tuple[DataArray, Dataset]:
        """## Simulate the instrument observation of a source spectrum.

        ### Args:
            - `source_wl (np.ndarray)`: Source spectrum wavelengths in Angstrom.
            - `source_i (np.ndarray)`: Source spectrum intensities in W/m^2/Anstrom.
            - `camera (MisCamera)`: Throughput and detector specifications. See `MisCamera`.
            - `alpha (Optional[Numeric], optional)`: Grating angle in degrees. Defaults to None.
            - `report (bool, optional)`: Report operations during calculations. Defaults to True.
            - `use_c (bool, optional)`: Use the faster C library for calculations. Defaults to True.

        ### Note:
        - The C library is faster for large datasets, but may not be available on all platforms.
        - The C library, additionally, requires that the upper and lower bounds of the wavelength ranges that are fed to the intensity integral are both monotonic. This is not a requirement for the Python implementation.

        ### Returns:
            - `Tuple[DataArray, Dataset]`: The first element is the intensity map, in electrons. The second element is a dataset containing the intensity, wavelength map, resolution map, and order map for each slit.
        """
        INV_PLANK_CONST = 1 / 6.62607015e-24  # adjusted for Angstrom
        SPEED_LIGHT = 299792458

        def calc_intensity(l, u):
            idx = np.where((source_wl >= l) & (source_wl <= u))
            return np.sum(source_i[idx])*(u-l)

        def report_print(show: bool, msg: str, end: str = '\n'):
            if show:
                print(msg, end=end)

        if alpha is not None:
            self.alpha = alpha

        # scale the pixel size to the mosaic coordinate
        dx = abs(camera.pixel_size / camera.scale)
        # in mosaic coordinates, goes from right to left, origin at bottom-left corner
        beta_grid = np.arange(0, -self.mosaic.width - dx, -dx)
        # in mosaic coordinates, goes from bottom to top
        gamma_grid = np.arange(0, self.mosaic.height + dx, dx)
        # print(f'Grid size: {len(beta_grid)}x{len(gamma_grid)}')
        beta_mesh, _ = np.meshgrid(beta_grid, gamma_grid)
        # print(f'Mesh size: {beta_mesh.shape[0]}x{beta_mesh.shape[1]}')
        intensities = np.zeros(beta_mesh.shape, dtype=float)
        intensities = DataArray(intensities, coords={
            'gamma': gamma_grid, 'beta': beta_grid}, dims=['gamma', 'beta'])
        prods: Dict[str, Optional[Dataset]] = {}
        for slit in self.slits.keys():
            prod = self._grating_product_sim(
                slit, self.alpha, beta_grid, gamma_grid)
            prods[slit] = prod

        valid_keys = [k if v is not None else None for k,
                      v in self.slits.items()]
        valid_keys = list(filter(lambda x: x is not None, valid_keys))
        extra_maps = Dataset(
            {
                'intensity': (['gamma', 'beta', 'slit'], np.zeros((*intensities.values.shape, len(valid_keys)), dtype=float)),
                'wavelength': (['gamma', 'beta', 'slit'], np.full((*intensities.values.shape, len(valid_keys)), np.nan, dtype=float)),
                'resolution': (['gamma', 'beta', 'slit'], np.full((*intensities.values.shape, len(valid_keys)), np.nan, dtype=float)),
                'order': (['gamma', 'beta', 'slit'], np.full((*intensities.values.shape, len(valid_keys)), np.nan, dtype=float)),
            },
            coords={
                'gamma': gamma_grid,
                'beta': beta_grid,
                'slit': valid_keys,
            },
            attrs={
                'alpha': self.alpha,
                'gmin': [prods[k].attrs['gmin'] for k in valid_keys],
                'gmax': [prods[k].attrs['gmax'] for k in valid_keys],
            }
        )

        dark_rate = 0
        if camera.dark_current is not None and camera.dark_current > 0:  # dark current
            dark_rate = camera.dark_current * camera.exposure

        for window in self.mosaic.windows:
            beta_range = window.get_xrange()
            gamma_range = window.get_yrange()
            report_print(report,
                         f'Window {window.name}: β ({beta_range[0]:.2f}, {beta_range[1]:.2f}), γ ({(gamma_range[0]):.2f}, {gamma_range[1]:.2f})')
            for skey, slit in self.slits.items():
                prod = prods[skey]
                if prod is None:
                    continue
                prod: Dataset = prod
                # it is guaranteed to be not None at this point
                props: Dataset = extra_maps.sel(slit=skey)
                wl_range = []
                for r in window.ranges:
                    if slit.ranges is not None and len(slit.ranges) > 0:
                        wl_range += common_range(slit.ranges, r)
                    else:
                        wl_range.append(r)
                for r in wl_range:
                    rmin = r[0]
                    rmax = r[1]
                    grange = common_range(
                        [(prod.attrs["gmin"], prod.attrs["gmax"])], gamma_range)
                    if len(grange) != 1:
                        if len(grange) > 1:
                            warnings.warn(
                                f'Gamma range for slit {skey} is outside the window range {window.name}: Mosaic: {gamma_range}, Slit: ({prod.attrs["gmin"]}, {prod.attrs["gmax"]}).')
                        continue
                    grange = grange[0]

                    bmin = np.min(beta_range)
                    bmax = np.max(beta_range)
                    gmin = np.min(grange)
                    gmax = np.max(grange)
                    prod_s = prod.sel(beta=slice(*beta_range),
                                      gamma=slice(*grange))
                    props_s = props.sel(beta=slice(*beta_range),
                                        gamma=slice(*grange))
                    intensities_s = intensities.sel(
                        beta=slice(*beta_range), gamma=slice(*grange))

                    n1_min, n1_max = sign_ceil(
                        np.nanmin(prod_s.grating_product.values / rmin)), sign_floor(np.nanmax(prod_s.grating_product.values / rmin))
                    n2_min, n2_max = sign_ceil(
                        np.nanmin(prod_s.grating_product.values / rmax)), sign_floor(np.nanmax(prod_s.grating_product.values / rmax))
                    n_min = int(min(n1_min, n2_min))
                    n_max = int(max(n1_max, n2_max))
                    if n_min > n_max:
                        n_min, n_max = n_max, n_min

                    report_print(report,
                                 f'\tSlit {skey}: λ ({rmin:.0f}, {rmax:.0f}), γ ({gmin:.2f}, {gmax:.2f}), β ({bmin:.2f}, {bmax:.2f}), β (prod) ({prod_s.beta.values[0]:.2f}, {prod_s.beta.values[-1]:.2f}), γ (prod) ({prod_s.gamma.values[0]:.2f}, {prod_s.gamma.values[-1]:.2f}) Orders ({n_min}, {n_max})')

                    for n in range(n_min, n_max + 1):
                        if n == 0:
                            continue
                        lam = prod_s.grating_product.values / n
                        dlam = prod_s.d_nx.values / n / 2
                        report_print(report,
                                     f'\t\tλ Valid: ({rmin:.2f}, {rmax:.2f}), Calculated: ({np.nanmin(lam):.2f}, {np.nanmax(lam):.2f}), Order {n}', end=', ')
                        sys.stdout.flush()
                        rvalid = np.where((lam >= rmin) & (lam <= rmax))
                        if len(rvalid[0]) == 0:
                            report_print(report, 'No valid wavelengths.')
                            continue
                        props_s.order.values[rvalid] = n
                        props_s.wavelength.values[rvalid] = lam[rvalid]
                        props_s.resolution.values[rvalid] = lam[rvalid] / \
                            dlam[rvalid]
                        # Interpolate QE curve
                        if camera.qe_curve is None:
                            qe = 1
                        else:
                            qe = np.interp(
                                lam[rvalid], camera.qe_curve[0], camera.qe_curve[1])
                        lower = lam[rvalid] - dlam[rvalid]
                        upper = lam[rvalid] + dlam[rvalid]
                        llower: np.ndarray = np.minimum(lower, upper)
                        uupper: np.ndarray = np.maximum(lower, upper)
                        if use_c:
                            sortargs = np.argsort(llower)
                            llower.sort()
                            uupper.sort()
                            start = perf_counter_ns()
                            # intensity = np.vectorize(
                            #     calc_intensity)(llower, uupper)
                            intensity = multi_integrate(
                                source_wl, source_i, llower, uupper)
                            intensity = unsort(intensity, sortargs)
                            end = perf_counter_ns()
                        else:
                            start = perf_counter_ns()
                            intensity = np.vectorize(
                                calc_intensity)(llower, uupper)
                            end = perf_counter_ns()
                        report_print(report,
                                     f'O({llower.size}): {(end - start)*1e-6:.3f} ms')
                        # intensity[rignr] = 0
                        # intensity = intensity.reshape(lam.shape)
                        # amount of energy in Joules
                        intensity = (intensity * camera.aperture *
                                     1e-6 * camera.exposure)
                        intensity = intensity * \
                            lam[rvalid] * INV_PLANK_CONST / \
                            SPEED_LIGHT  # convert to photons
                        intensity = intensity * qe * camera.optical_efficiency  # apply QE and optical efficiency
                        if camera.readout_noise is not None and camera.readout_noise > 0:  # readout noise
                            intensity += np.random.poisson(
                                0, camera.readout_noise, intensity.shape)
                        intensity += dark_rate  # dark current
                        intensities_s.values[rvalid] += intensity
                        props_s.intensity.values[rvalid] += intensity
                        # intensities[midx] += (intensity * (dx * dx) * 1e-6 * camera.exposure * qe)
            report_print(report, f'Window {window.name} processed.')
        # return intensities
        intensities.values.clip(0, camera.well_depth, out=intensities.values)
        extra_maps.intensity.values.clip(
            0, camera.well_depth, out=extra_maps.intensity.values)
        return (intensities, extra_maps)

    def intensity_plot(self, intensities: DataArray, wavelengths: List[int | MisFeatures] = None, *, default_style={'ls': '-', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}, cmap: str = 'bone', **fig_kwargs):
        fig, ax = self._plot_lines(False, self.alpha, wavelengths, mode='Mosaic',
                                   default_style=default_style, labels=True, labelcolor='w', fig_kwargs=fig_kwargs)
        fig: plt.Figure = fig
        ax: plt.Axes = ax
        im = ax.imshow(intensities.values, origin='lower', extent=[
                       intensities.beta.values[0], intensities.beta.values[-1], intensities.gamma.values[0], intensities.gamma.values[-1]], cmap=cmap)
        # fig.subplots_adjust(bottom=0.7)
        cax = fig.add_axes([0.1, 0, 0.8, 0.01])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label('Intensity (e$^-$)')
        cbar.formatter.set_useMathText(True)
        return fig, ax, cax

    def order_map(self, extra_maps: Dataset, wavelengths: List[int | MisFeatures] = None, *, default_style={'ls': '-', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}, **fig_kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """## Plot the different orders illuminating different sections of the mosaic for all slits.
        ### Args:
            - `extra_maps (Dataset)`: Dataset containing the intensity, wavelength map, resolution map, and order map for each slit.
            - `wavelengths (List[int  |  MisFeatures], optional)`: Wavelength features of interest. Defaults to None. If the object was used with a set of features previously, this argument is not required.
            - `default_style (dict, optional)`: Default plot profile for the spectral features. Defaults to {'ls': '-', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}.

        ### Returns:
            - `Tuple[plt.Figure, plt.Axes]`: Created figure and axes objects.
        """
        output = []
        for k in extra_maps.slit.values:
            v = extra_maps.sel(slit=k)
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

        def sortby(x):
            return x[-1]

        output.sort(key=sortby, reverse=True)
        fig, ax = self._plot_lines(False, self.alpha, wavelengths, mode='Mosaic',
                                   default_style=default_style, labels=False, labelcolor='k', fig_kwargs=fig_kwargs)
        cmap = plt.cm.gist_rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=len(output) - 1)
        colors = [cmap(norm(i)) for i in range(len(output))]
        import random
        random.shuffle(colors)
        for kidx, v in enumerate(output):
            k, order, bb, gg, lidx = v
            color = colors[kidx]
            ax.plot(bb, gg, color=color, zorder=kidx+5)
            ax.axhline(0, color=color, lw=0.5, zorder=0, label=f'{k}: {int(order)}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        return fig, ax

    def order_map_slit(self, extra_maps: Dataset, slit: str, wavelengths: List[int | MisFeatures] = None, *, default_style={'ls': '-', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}, **fig_kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """## Plot the order map for a specific slit.

        ### Args:
            - `extra_maps (Dataset)`: Dataset containing the intensity, wavelength map, resolution map, and order map for each slit.
            - `slit (str)`: Slit key.
            - `wavelengths (List[int  |  MisFeatures], optional)`: Pass the spectral features of interest. Defaults to None. If the object was used with a set of features previously, this argument is not required.
            - `default_style (dict, optional)`: Default line plot style. Defaults to {'ls': '-', 'lw': 0.5, 'ms': 0.2, 'color': 'black'}.

        ### Raises:
            - `ValueError`: Slit not found in the dataset.

        ### Returns:
            - `Tuple[plt.Figure, plt.Axes]`: Created figure and axes objects.
        """
        if slit not in extra_maps.slit.values:
            raise ValueError(f'Slit {slit} not found in the dataset.')
        v = extra_maps.sel(slit=slit)
        orders = np.unique(v['order'].values)
        output = []
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
            output.append((order, bb, gg, len(lidx[0])))

        def sortby(x):
            return x[-1]
        
        def filterby(x):
            if x[-1] < 200:
                return False
            return True
        output = list(filter(filterby, output))
        output.sort(key=sortby, reverse=True)

        fig, ax = self._plot_lines(False, self.alpha, wavelengths, mode='Mosaic',
                                   default_style=default_style, labels=False, labelcolor='k', fig_kwargs=fig_kwargs)
        cmap = plt.cm.gist_rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=len(output) - 1)
        colors = [cmap(norm(i)) for i in range(len(output))]
        import random
        random.shuffle(colors)
        for kidx, v in enumerate(output):
            order, bb, gg, lidx = v
            color = colors[kidx]
            ax.plot(bb, gg, color=color, zorder=kidx+5)
            ax.axhline(0, color=color, lw=0.5, zorder=0, label=f'{slit}: {int(order)}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        return fig, ax
# %%
