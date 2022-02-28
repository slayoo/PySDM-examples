from pystrict import strict
import numpy as np
from PySDM.physics.constants import si
from PySDM.initialisation.spectra import Gamma
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM.formulae import Formulae
from PySDM.physics.constants_defaults import rho_w

@strict
class Settings:

    def __init__(self):
        self.formulae = Formulae()
        self.n_sd = 2**8
        self.n_part = 1e4 / si.cm**3
        self.theta = 0.33e-9 * si.g / rho_w
        self.k = 1
        self.dv = 1 * si.m**3
        self.norm_factor = self.n_part * self.dv
        self.dt = 1 * si.seconds
        self.adaptive = False
        self.seed = 44
        self._steps = [i for i in range(60)] #[0, 60]
        self.kernel = Golovin(b=2000 * si.cm**3 / si.g / si.s * rho_w)
        self.coal_effs = [ConstEc(Ec=0.8), ConstEc(Ec=0.9),ConstEc(Ec=1.0)]
        self.fragmentation = AlwaysN(n=3)
        self.break_eff = ConstEb(1.0) # no "bouncing"
        self.spectrum = Gamma(norm_factor=self.norm_factor, k=self.k, theta=self.theta)
        # self.radius_bins_edges = np.logspace(
        #     np.log10(1 * si.um),
        #     np.log10(1000 * si.um),
        #     num=128, endpoint=True
        # )
        # self.radius_range = [0 * si.um, 1e6 * si.um]
        self.rho = rho_w

    @property
    def output_steps(self):
        return [int(step/self.dt) for step in self._steps]
