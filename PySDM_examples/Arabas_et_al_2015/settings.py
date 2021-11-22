from typing import Iterable
from pystrict import strict
from PySDM.physics import si
from PySDM.backends.impl_numba.conf import JIT_FLAGS
from PySDM_examples.Morrison_and_Grabowski_2007.strato_cumulus import StratoCumulus


@strict
class Settings(StratoCumulus):
    def __dir__(self) -> Iterable[str]:
        return 'dt', 'grid', 'size', 'n_spin_up', 'versions', 'steps_per_output_interval',\
               'formulae', 'initial_dry_potential_temperature_profile',\
               'initial_vapour_mixing_ratio_profile'

    def __init__(self, fastmath: bool = JIT_FLAGS['fastmath']):
        super().__init__(
            fastmath,
            rho_w_max=.6 * si.metres / si.seconds * (si.kilogram / si.metre ** 3)
        )

        self.grid = (25, 25)
        self.size = (1500 * si.metres, 1500 * si.metres)

        # output steps
        self.simulation_time = 90 * si.minute
        self.dt = 5 * si.second
        self.spin_up_time = 1 * si.hour
