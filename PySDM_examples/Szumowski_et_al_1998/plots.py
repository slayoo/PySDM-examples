import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PySDM import products
from PySDM.physics import constants as const


class _Plot:
    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.ax.set_title(' ')


class _ImagePlot(_Plot):
    line_args = {'color': 'red', 'alpha': .75, 'linestyle': ':', 'linewidth': 5}

    def __init__(self, fig, ax, grid, size, product, show=False, lines=False, cmap='YlGnBu'):
        super().__init__(fig, ax)
        self.nans = np.full(grid, np.nan)

        self.dx = size[0] / grid[0]
        self.dz = size[1] / grid[1]

        xlim = (0, size[0])
        zlim = (0, size[1])

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(zlim)

        if lines:
            self.lines = {'X': [None]*2, 'Z': [None]*2}
            self.lines['X'][0] = plt.plot([-1] * 2, zlim, **self.line_args)[0]
            self.lines['Z'][0] = plt.plot(xlim, [-1] * 2, **self.line_args)[0]
            self.lines['X'][1] = plt.plot([-1] * 2, zlim, **self.line_args)[0]
            self.lines['Z'][1] = plt.plot(xlim, [-1] * 2, **self.line_args)[0]

        product_range, product_scale = {
            products.WaterMixingRatio: ((0, 1), 'linear'),
            products.TotalParticleSpecificConcentration: ((20, 50), 'linear'),
            products.TotalParticleConcentration: ((20, 50), 'linear'),
            products.SuperDropletCount: ((0, 10), 'linear'),
            products.ParticlesVolumeSpectrum: ((20, 50), 'linear'),
            products.AerosolConcentration: ((1e0, 1e2), 'linear'),
            products.CloudDropletConcentration: ((1e0, 1e2), 'linear'),
            products.DrizzleConcentration: ((1e-3, 1e1), 'log'),
            products.ParticleMeanRadius: ((1, 25), 'linear'),
            products.CloudDropletEffectiveRadius: ((0, 25), 'linear'),
            products.AerosolSpecificConcentration: ((0, 3e2), 'linear'),
            products.WaterVapourMixingRatio: ((5, 7.5), 'linear'),
            products.Temperature: ((275, 305), 'linear'),
            products.RelativeHumidity: ((75, 105), 'linear'),
            products.Pressure: ((90000, 100000), 'linear'),
            products.DryAirPotentialTemperature: ((275, 300), 'linear'),
            products.DryAirDensity: ((0.95, 1.3), 'linear'),
            products.PeakSupersaturation: ((-1, 1), 'linear'),
            products.RipeningRate: ((1e-1, 1e1), 'log'),
            products.ActivatingRate: ((1e-1, 1e1), 'log'),
            products.DeactivatingRate: ((1e-1, 1e1), 'log'),
            products.CollisionRateDeficit: ((0, 1e10), 'linear'),
            products.CollisionRate: ((0, 1e10), 'linear'),
            products.CondensationTimestepMin: ((.01, 10), 'log'),
            products.CondensationTimestepMax: ((.01, 10), 'log'),
            products.CoalescenceTimestepMin: ((.01, 10), 'log'),
            products.IceWaterContent: ((.0001, .001), 'linear'),
            products.ParticlesConcentration: ((0, 1e4), 'linear')
        }[product.__class__]
        data = np.full_like(self.nans, product_range[0])
        label = f"{product.description} [{product.unit}]"

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Z [m]')

        self.im = self.ax.imshow(
            self._transpose(data),
            origin='lower',
            extent=(*xlim, *zlim),
            cmap=cmap,
            norm=(
                matplotlib.colors.LogNorm() if product_scale == 'log' and np.isfinite(data).all()
                else None
            )
        )
        plt.colorbar(self.im, ax=self.ax).set_label(label)
        self.im.set_clim(vmin=product_range[0], vmax=product_range[1])
        if show:
            plt.show()

    @staticmethod
    def _transpose(data):
        if data is not None:
            return data.T

    def update(self, data,  step):
        data = self._transpose(data)
        if data is not None:
            self.im.set_data(data)
            self.ax.set_title(
                f"min:{np.nanmin(data): .3g}    max:{np.nanmax(data): .3g}    t/dt_out:{step: >6}"
            )

    def update_lines(self, focus_x, focus_z):
        self.lines['X'][0].set_xdata(x=focus_x[0] * self.dx)
        self.lines['Z'][0].set_ydata(y=focus_z[0] * self.dz)
        self.lines['X'][1].set_xdata(x=focus_x[1] * self.dx)
        self.lines['Z'][1].set_ydata(y=focus_z[1] * self.dz)


class _SpectrumPlot(_Plot):

    def __init__(self, r_bins, initial_spectrum_per_mass_of_dry_air, show=True):
        super().__init__(*plt.subplots(1, 1))
        self.ax.set_xlim(np.amin(r_bins), np.amax(r_bins))
        self.ax.set_xlabel("particle radius [μm]")
        self.ax.set_ylabel("specific concentration density [mg$^{-1}$ μm$^{-1}$]")
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_ylim(1, 5e3)
        self.ax.grid(True)
        vals = initial_spectrum_per_mass_of_dry_air.size_distribution(r_bins * const.si.um)
        const.convert_to(vals, const.si.mg**-1 / const.si.um)
        self.ax.plot(r_bins, vals, label='spectrum sampled at t=0')
        self.spec_wet = self.ax.step(r_bins, np.full_like(r_bins, np.nan),
                                     label='binned super-particle wet sizes')[0]
        self.spec_dry = self.ax.step(r_bins, np.full_like(r_bins, np.nan),
                                     label='binned super-particle dry sizes')[0]
        self.ax.legend()
        if show:
            plt.show()

    def update_wet(self, data, step):
        self.spec_wet.set_ydata(data)
        self.ax.set_title(f"t/dt_out:{step}")

    def update_dry(self, dry):
        self.spec_dry.set_ydata(dry)


class _TimeseriesPlot(_Plot):

    def __init__(self, fig, ax, times, show=True):
        super().__init__(fig, ax)
        self.ax.set_xlim(0, times[-1])
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("rainfall [mm/day]")
        self.ax.set_ylim(0, 1e-1)
        self.ax.grid(True)
        self.ydata = np.full_like(times, np.nan, dtype=float)
        self.timeseries = self.ax.step(times, self.ydata, where='pre')[0]
        if show:
            plt.show()

    def update(self, data):
        if data is not None:
            self.ydata[0:len(data)] = data[:]
        else:
            self.ydata[:] = np.nan
        self.timeseries.set_ydata(self.ydata)


class _TemperaturePlot(_Plot):
    def __init__(self, T_bins, formulae, show=True):
        super().__init__(*plt.subplots(1, 1))
        self.formulae = formulae
        self.ax.set_xlim(np.amax(T_bins), np.amin(T_bins))
        self.ax.set_xlabel("temperature [K]")
        self.ax.set_ylabel("freezable fraction / cdf [1]")
        self.ax.set_ylim(-.05, 1.05)
        self.ax.grid(True)
        self.ax.plot(T_bins, self.formulae.freezing_temperature_spectrum.cdf(T_bins),
                     label=str(self.formulae.freezing_temperature_spectrum) + " (sampled at t=0)")
        P0 = .5
        T0 = 33
        bigg_cdf = lambda TS: np.exp(np.log(1-P0)*np.exp(T0-TS))
        self.ax.plot(T_bins, bigg_cdf(const.T0 - T_bins), label=f'Bigg 1953 for (TS_median={T0})')
        self.spec = self.ax.step(T_bins, np.full_like(T_bins, np.nan),
                                 label='binned super-particle attributes', where='mid')[0]
        self.ax.legend()
        if show:
            plt.show()

    def update(self, data, step):
        self.ax.set_title(f"t/dt_out:{step}")
        self.spec.set_ydata(data)
