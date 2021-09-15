from PySDM import Builder
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.physics import constants as const, Formulae
from PySDM.initialisation import spectral_sampling
from PySDM.initialisation.multiplicities import discretise_n
from PySDM.products import IceWaterContent
from PySDM.backends import CPU
import numpy as np


def simulation(*, seed, n_sd, dt, dv, spectrum, droplet_volume, multiplicity, J_het, total_time, number_of_real_droplets):
    formulae = Formulae(seed=seed)
    builder = Builder(n_sd=n_sd, backend=CPU, formulae=formulae)
    builder.set_environment(Box(dt=dt, dv=dv))
    builder.add_dynamic(Freezing(singular=False, J_het=J_het))

    if hasattr(spectrum, 's_geom') and spectrum.s_geom==1:
        _isa, _conc = np.full(n_sd, spectrum.m_mode), np.full(n_sd, multiplicity / dv)
    else:
        _isa, _conc = spectral_sampling.ConstantMultiplicity(spectrum).sample(n_sd)
    attributes = {
        'n': discretise_n(_conc * dv),
        'immersed surface area': _isa,
        'volume': np.full(n_sd, droplet_volume)
    }
    np.testing.assert_almost_equal(attributes['n'], multiplicity)
    products = [IceWaterContent(specific=False)]
    particulator = builder.build(attributes=attributes, products=products)

    cell_id = 0
    data = []
    for i in range(int(total_time / dt) + 1):
        particulator.run(0 if i == 0 else 1)

        ice_mass_per_volume = particulator.products['qi'].get()[cell_id]
        ice_mass = ice_mass_per_volume * dv
        ice_number = ice_mass / (const.rho_w * droplet_volume)
        unfrozen_fraction = 1 - ice_number / number_of_real_droplets
        data.append(unfrozen_fraction)
    return data
