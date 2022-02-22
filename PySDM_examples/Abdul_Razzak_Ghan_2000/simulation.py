import numpy as np
from PySDM.environments import Parcel
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.initialisation import discretise_multiplicities, equilibrate_wet_radii
from PySDM.initialisation.spectra import Sum
import PySDM.products as PySDM_products
from PySDM_examples.utils import BasicSimulation

class Magick:
    def register(self, builder):
        builder.request_attribute("critical supersaturation")

    def __call__(self):
        pass

class Simulation:
    def __init__(self, settings, products=None):
        env = Parcel(dt=settings.dt,
                     mass_of_dry_air=settings.mass_of_dry_air,
                     p0=settings.p0,
                     q0=settings.q0,
                     T0=settings.T0,
                     w=settings.w)
        n_sd = settings.n_sd_per_mode * len(settings.aerosol.aerosol_modes_per_cc)
        builder = Builder(n_sd=n_sd, backend=CPU(formulae=settings.formulae))
        builder.set_environment(env)

        attributes = {
            'volume':np.empty(n_sd),
            'n': np.ndarray(n_sd),
            'dry volume':np.empty(n_sd),
            'kappa times dry volume':np.empty(n_sd),
        }
        for i, mode in enumerate(settings.aerosol.aerosol_modes_per_cc):
            r_dry, concentration = settings.spectral_sampling(
                spectrum=mode['spectrum']).sample(settings.n_sd_per_mode)
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            specific_concentration = concentration / settings.formulae.constants.rho_STP
            chunk = slice(i * settings.n_sd_per_mode, (i+1) * settings.n_sd_per_mode)
            attributes['n'][chunk] = specific_concentration * env.mass_of_dry_air
            attributes['dry volume'][chunk] = v_dry
            attributes['kappa times dry volume'][chunk] = mode['kappa']*v_dry
            r_wet = equilibrate_wet_radii(r_dry, env, mode['kappa']*v_dry)
            attributes['volume'][chunk] = builder.formulae.trivia.volume(radius=r_wet)
        
        # for attribute in attributes.values():
        #     assert attribute.shape[0] == n_sd

#         dv = settings.mass_of_dry_air / settings.rho0
#         np.testing.assert_approx_equal(
#             np.sum(attributes['n']) / dv,
#             Sum(tuple(
#                 settings.aerosol.aerosol_modes_per_cc[i]['spectrum']
#                 for i in range(len(settings.aerosol.aerosol_modes_per_cc))
#             )).norm_factor,
#             significant=5
#         )

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())
        builder.add_dynamic(Magick())

        products = products or (
            PySDM_products.WaterMixingRatio(name = "ql", unit = "g/kg", radius_range = (0.0, np.inf)), 
            PySDM_products.PeakSupersaturation(name = "S max"),                                                                                                                          
            PySDM_products.RipeningRate(name = "Ripe Rate"), 
            PySDM_products.DeactivatingRate(name = "Deact Rate"), 
            PySDM_products.ActivatingRate(name = "Act Rate"), 
            PySDM_products.ParcelDisplacement(name = "z"),
            PySDM_products.ParticleSizeSpectrumPerMass(
                name = "Particle Size Spectrum Per Mass", 
                radius_bins_edges=settings.wet_radius_bins_edges, unit= "1/um/mg"),
        )

        self.particulator = builder.build(attributes=attributes, products=products)
        self.settings = settings

    def _save(self, output):
        for k, v in self.particulator.products.items():
            value = v.get()
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value[0]
            output[k].append(value)

    def run(self):
        output = {k: [] for k in self.particulator.products}
        for step in self.settings.output_steps:
            self.particulator.run(step - self.particulator.n_steps)
            self._save(output)
        return output
    