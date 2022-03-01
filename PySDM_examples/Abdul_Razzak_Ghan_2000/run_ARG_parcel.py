import numpy as np

from PySDM import Formulae
from PySDM import Builder, products as PySDM_products
from PySDM.physics import si
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.backends import CPU
from PySDM.backends.impl_numba.test_helpers import bdf
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel

from PySDM_examples.Abdul_Razzak_Ghan_2000.aerosol import AerosolARG

class Magick:
    def register(self, builder):
        builder.request_attribute("critical supersaturation")

    def __call__(self):
        pass

def run_parcel(w, sol2, N2, rad2, n_sd_per_mode):
    products = (
     PySDM_products.WaterMixingRatio(unit="g/kg", name="ql"),
     PySDM_products.PeakSupersaturation(name="S max"),
     PySDM_products.AmbientRelativeHumidity(name="RH"),
     PySDM_products.ParcelDisplacement(name="z")
    )
    
    formulae = Formulae()
    const = formulae.constants
    RH0, T0, p0 = 1.0, 294, 1e5
    pv0 = RH0 * formulae.saturation_vapour_pressure.pvs_Celsius(T0 - const.T0)
    q0 = const.eps * pv0 / (p0 - pv0)

    env = Parcel(
     dt=2 * si.s,
     mass_of_dry_air=1e3 * si.kg,
     p0=p0 * si.Pa,
     q0=q0 * si.kg / si.kg,
     w=w * si.m / si.s,
     T0=T0 * si.K
    )

    aerosol = AerosolARG(M2_sol=sol2, M2_N=N2, M2_rad=rad2)
    n_steps = 50
    n_sd = n_sd_per_mode * len(aerosol.aerosol_modes)

    builder = Builder(backend=CPU(), n_sd=n_sd)
    builder.set_environment(env)
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())
    builder.add_dynamic(Magick())

    attributes = {k: np.empty(0) for k in ('dry volume', 'kappa times dry volume', 'n')}
    for i, mode in enumerate(aerosol.aerosol_modes):
        kappa, spectrum = mode["kappa"], mode["spectrum"]
        r_dry, concentration = ConstantMultiplicity(spectrum).sample(n_sd_per_mode)
        v_dry = builder.formulae.trivia.volume(radius=r_dry)
        specific_concentration = concentration / builder.formulae.constants.rho_STP
        attributes['n'] = np.append(attributes['n'], specific_concentration * env.mass_of_dry_air)
        attributes['dry volume'] = np.append(attributes['dry volume'], v_dry)
        attributes['kappa times dry volume'] = np.append(attributes['kappa times dry volume'], v_dry * kappa)
    
    r_wet = equilibrate_wet_radii(
        r_dry=builder.formulae.trivia.radius(volume=attributes['dry volume']),
        environment=env,
        kappa_times_dry_volume=attributes['kappa times dry volume'],
    )
    attributes['volume'] = builder.formulae.trivia.volume(radius=r_wet)

    particulator = builder.build(attributes, products=products)
    bdf.patch_particulator(particulator)

    output = {product.name: [] for product in particulator.products.values()}
    output_attributes = {'n': tuple([] for _ in range(particulator.n_sd)),
                        'volume': tuple([] for _ in range(particulator.n_sd)),
                        'critical volume': tuple([] for _ in range(particulator.n_sd)),
                        'critical supersaturation': tuple([] for _ in range(particulator.n_sd))}

    for _ in range(n_steps):
        particulator.run(steps=1)
        for product in particulator.products.values():
            value = product.get()
            output[product.name].append(value[0])
        for key, attr in output_attributes.items():
            attr_data = particulator.attributes[key].to_ndarray()
            for drop_id in range(particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])
    
    error = np.zeros(len(aerosol.aerosol_modes))
    activated_fraction_S = np.zeros(len(aerosol.aerosol_modes))
    activated_fraction_V = np.zeros(len(aerosol.aerosol_modes))
    for j, mode in enumerate(aerosol.aerosol_modes):
        activated_drops_j_S = 0
        activated_drops_j_V = 0
        RHmax = np.nanmax(np.asarray(output['RH']))
        for i, volume in enumerate(output_attributes['volume']):
            if j*n_sd_per_mode <= i < (j+1)*n_sd_per_mode:
                if output_attributes['critical supersaturation'][i][-1] < RHmax:
                    activated_drops_j_S += output_attributes['n'][i][-1]
                if output_attributes['critical volume'][i][-1] < volume[-1]:
                    activated_drops_j_V += output_attributes['n'][i][-1]
        Nj = np.asarray(output_attributes['n'])[j*n_sd_per_mode:(j+1)*n_sd_per_mode, -1]
        max_multiplicity_j = np.max(Nj)
        sum_multiplicity_j = np.sum(Nj)
        error[j] = max_multiplicity_j / sum_multiplicity_j
        activated_fraction_S[j] = activated_drops_j_S / sum_multiplicity_j
        activated_fraction_V[j] = activated_drops_j_V / sum_multiplicity_j
    
    return output, output_attributes, aerosol, activated_fraction_S, activated_fraction_V, error