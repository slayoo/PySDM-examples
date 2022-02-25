import numpy as np
from PySDM.environments import Parcel
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.initialisation import discretise_multiplicities, equilibrate_wet_radii
from PySDM.initialisation.spectra import Sum
import PySDM.products as PySDM_products
from PySDM_examples.utils import BasicSimulation
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity

class Magick:
    def register(self, builder):
        builder.request_attribute("critical supersaturation")

    def __call__(self):
        pass

class Simulation(BasicSimulation):
    def __init__(self, settings, products=None):
        print(settings.p0, settings.T0, settings.q0)
        env = Parcel(dt=settings.dt,
                     mass_of_dry_air=settings.mass_of_dry_air,
                     p0=settings.p0,
                     q0=settings.q0,
                     T0=settings.T0,
                     w=settings.w)
        n_sd = settings.n_sd_per_mode * len(settings.aerosol.aerosol_modes)
        builder = Builder(n_sd=n_sd, backend=CPU(formulae=settings.formulae))
        builder.set_environment(env)

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
        
        volume = env.mass_of_dry_air / settings.rho0
        attributes = {k: np.empty(0) for k in ('dry volume', 'kappa times dry volume', 'n')}
        for i, mode in enumerate(settings.aerosol.aerosol_modes):
            kappa, spectrum = mode["kappa"], mode["spectrum"]
            sampling = ConstantMultiplicity(spectrum)
            r_dry, n_per_volume = sampling.sample(settings.n_sd_per_mode)
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            attributes['n'] = np.append(attributes['n'], n_per_volume * volume)
            attributes['dry volume'] = np.append(attributes['dry volume'], v_dry)
            attributes['kappa times dry volume'] = np.append(
                attributes['kappa times dry volume'], v_dry * kappa)
        
        r_wet = equilibrate_wet_radii(
            r_dry=settings.formulae.trivia.radius(volume=attributes['dry volume']),
            environment=env,
            kappa_times_dry_volume=attributes['kappa times dry volume'],
        )
        
        attributes['volume'] = settings.formulae.trivia.volume(radius=r_wet)

        super().__init__(particulator=builder.build(attributes=attributes, products=products))

        self.output_attributes = {'volume': tuple([] for _ in range(self.particulator.n_sd)),
                                 'critical volume': tuple([] for _ in range(self.particulator.n_sd)),
                                 'critical supersaturation': tuple([] for _ in range(self.particulator.n_sd)),
                                 'n': tuple([] for _ in range(self.particulator.n_sd)),}
        self.settings = settings

        self.__sanity_checks(attributes, volume)

    def __sanity_checks(self, attributes, volume):
        for attribute in attributes.values():
            assert attribute.shape[0] == self.particulator.n_sd
        np.testing.assert_approx_equal(
            np.sum(attributes['n']) / volume,
            np.sum(mode['spectrum'].norm_factor for mode in self.settings.aerosol.aerosol_modes),
            significant=4
        )

    def _save(self, output):
        for key, attr in self.output_attributes.items():
            attr_data = self.particulator.attributes[key].to_ndarray()
            for drop_id in range(self.particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])
        super()._save(output)

    def run(self):
        output_products = super()._run(self.settings.nt, self.settings.steps_per_output_interval)
        return {
            'products': output_products,
            'attributes': self.output_attributes
        }
