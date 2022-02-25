import numpy as np
from pystrict import strict
from PySDM import Formulae
from PySDM.initialisation.sampling import spectral_sampling as spec_sampling
from PySDM.physics import si
from PySDM_examples.Abdul_Razzak_Ghan_2000.aerosol import Aerosol


@strict
class Settings:
    def __init__(self, 
                 nsteps: int, 
                 n_sd_per_mode: int,
                 T0: float,
                 aerosol: Aerosol,
                 spectral_sampling: type(spec_sampling.SpectralSampling),
                 ):

        self.n_sd_per_mode = n_sd_per_mode
        self.formulae = Formulae()

        const = self.formulae.constants
        self.aerosol = aerosol
        self.spectral_sampling = spectral_sampling

        self.dt = 1.0 * si.s
        self.t_max = nsteps * self.dt
        self.output_interval = self.dt * 10

        self.w = 0.5 * si.m / si.s
        self.g = 9.81 * si.m / si.s**2

        self.p0 = 1000 * si.hPa
        self.T0 = T0 * si.K
        RH0 = 1.0
        pv0 = RH0 * self.formulae.saturation_vapour_pressure.pvs_Celsius(
            self.T0 - const.T0
        )
        self.q0 = const.eps * pv0 / (self.p0 - pv0)

        self.cloud_radius_range = (
                .5 * si.um,
                25 * si.um
        )

        self.mass_of_dry_air = 1e3 * si.kg

        self.wet_radius_bins_edges = np.logspace(
            np.log10(10 * si.nm),
            np.log10(1000 * si.nm),
            50+1,
            endpoint=True
        )

    @property
    def rho0(self):
        const = self.formulae.constants
        rhod0 = self.formulae.trivia.p_d(self.p0, self.q0) / self.T0 / const.Rd
        return rhod0 * (1 + self.q0)

    @property
    def nt(self):
        nt = self.t_max / self.dt
        assert nt == int(nt)
        return int(nt)

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.nt + 1, self.steps_per_output_interval)
