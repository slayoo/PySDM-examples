import numba
import scipy
import numpy, numpy as np
import PyMPDATA
import PySDM
from PySDM.physics import si, Formulae, constants as const
from PySDM.dynamics import condensation, coalescence
from PySDM.physics.coalescence_kernels import Geometric
from PySDM.physics import spectra


class Common:
    def __init__(self, fastmath):
        self.formulae = Formulae(condensation_coordinate='VolumeLogarithm', fastmath=fastmath)
        self.g = const.g_std

        self.condensation_rtol_x = condensation.default_rtol_x
        self.condensation_rtol_thd = condensation.default_rtol_thd
        self.condensation_adaptive = True
        self.condensation_substeps = -1
        self.condensation_dt_cond_range = condensation.default_cond_range
        self.condensation_schedule = condensation.default_schedule

        self.coalescence_adaptive = True
        self.coalescence_dt_coal_range = coalescence.default_dt_coal_range
        self.coalescence_optimized_random = True
        self.coalescence_substeps = 1
        self.kernel = Geometric(collection_efficiency=1)

        self.n_sd_per_gridbox = 20
        self.aerosol_radius_threshold = .5 * si.micrometre
        self.drizzle_radius_threshold = 25 * si.micrometre
        self.r_bins_edges = np.logspace(np.log10(0.001 * si.micrometre),
                                        np.log10(100 * si.micrometre),
                                        64, endpoint=True)
        self.T_bins_edges = np.linspace(const.T0-40, const.T0-20, 64, endpoint=True)
        self.output_interval = 1 * si.minute
        self.spin_up_time = 0

        self.mode_1 = spectra.Lognormal(
            norm_factor=60 / si.centimetre ** 3 / const.rho_STP,
            m_mode=0.04 * si.micrometre,
            s_geom=1.4
        )
        self.mode_2 = spectra.Lognormal(
          norm_factor=40 / si.centimetre**3 / const.rho_STP,
          m_mode=0.15 * si.micrometre,
          s_geom=1.6
        )
        self.spectrum_per_mass_of_dry_air = spectra.Sum((self.mode_1, self.mode_2))
        self.kappa = 1  # TODO #441!

        self.processes = {
            "particle advection": True,
            "fluid advection": True,
            "coalescence": True,
            "condensation": True,
            "sedimentation": True,
            "freezing": False,
            "PartMC piggy-backer": False
        }

        self.mpdata_iters = 2
        self.mpdata_iga = True
        self.mpdata_fct = True
        self.mpdata_tot = True

        key_packages = (PySDM, PyMPDATA, numba, numpy, scipy)
        try:
            import ThrustRTC
            key_packages.append(ThrustRTC)
        except:
            pass
        self.versions = {}
        for pkg in key_packages:
            try:
                self.versions[pkg.__name__] = pkg.__version__
            except AttributeError:
                pass
        self.versions = str(self.versions)

    @property
    def n_steps(self) -> int:
        return int(self.simulation_time / self.dt)  # TODO #413

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)

    @property
    def n_spin_up(self) -> int:
        return int(self.spin_up_time / self.dt)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.n_steps + 1, self.steps_per_output_interval)

    @property
    def n_sd(self):
        return self.grid[0] * self.grid[1] * self.n_sd_per_gridbox

    @property
    def initial_vapour_mixing_ratio_profile(self):
        return np.full(self.grid[-1], self.qv0)

    @property
    def initial_dry_potential_temperature_profile(self):
        return np.full(self.grid[-1], self.formulae.state_variable_triplet.th_dry(self.th_std0, self.qv0))