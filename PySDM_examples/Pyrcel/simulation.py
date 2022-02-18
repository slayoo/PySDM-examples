import numpy as np
from PySDM.environments import Parcel
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.spectra import Sum
import PySDM.products as PySDM_products
from PySDM_examples.utils import BasicSimulation


class Simulation(BasicSimulation):
    def __init__(self, settings, products=None):
        env = Parcel(dt=settings.dt,
                     mass_of_dry_air=settings.mass_of_dry_air,
                     p0=settings.p0,
                     q0=settings.q0,
                     T0=settings.T0,
                     w=settings.w)
        n_sd = sum(settings.n_sd_per_mode)
        builder = Builder(n_sd=n_sd, backend=CPU(formulae=settings.formulae))
        builder.set_environment(env)

        attributes = {
            'dry volume':np.empty(0),
            'dry volume organic':np.empty(0),
            'kappa times dry volume':np.empty(0),
            'n': np.ndarray(0)
        }
        for i,mode in enumerate(settings.aerosol.aerosol_modes_per_cc):
            r_dry, n_in_dv = settings.spectral_sampling(
                spectrum=mode['spectrum']).sample(settings.n_sd_per_mode[i])
            V = settings.mass_of_dry_air / settings.rho0
            N = n_in_dv * V
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            attributes['n'] = np.append(attributes['n'], N)
            attributes['dry volume'] = np.append(attributes['dry volume'], v_dry)
            attributes['dry volume organic'] = np.append(
                attributes['dry volume organic'], mode['f_org'] * v_dry)
            attributes['kappa times dry volume'] = np.append(
                attributes['kappa times dry volume'], v_dry * mode['kappa'])
        for attribute in attributes.values():
            assert attribute.shape[0] == n_sd

        np.testing.assert_approx_equal(
            np.sum(attributes['n']) / V,
            Sum(tuple(
                settings.aerosol.aerosol_modes_per_cc[i]['spectrum']
                for i in range(len(settings.aerosol.aerosol_modes_per_cc))
            )).norm_factor,
            #significant=5
            significant=4
        )
        r_wet = equilibrate_wet_radii(
            r_dry=settings.formulae.trivia.radius(volume=attributes['dry volume']),
            environment=env,
            kappa_times_dry_volume=attributes['kappa times dry volume'],
            f_org=attributes['dry volume organic'] / attributes['dry volume']
        )
        attributes['volume'] = settings.formulae.trivia.volume(radius=r_wet)
        del attributes['dry volume organic']

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        products = products or (
            PySDM_products.ParcelDisplacement(name='z'),
            PySDM_products.Time(name='t'),
            PySDM_products.PeakSupersaturation(unit='%', name='S_max'),
            PySDM_products.ParticleConcentration(name='n_c_cm3', unit='cm^-3',
                radius_range=settings.cloud_radius_range),
            PySDM_products.ParticleSizeSpectrumPerVolume(
                name='wet size spectrum',
                radius_bins_edges=settings.wet_radius_bins_edges),
            PySDM_products.ParticleSizeSpectrumPerVolume(
                name='dry size spectrum',
                radius_bins_edges=settings.dry_radius_bins_edges, dry=True),
        )

        particulator = builder.build(attributes=attributes, products=products)
        self.settings = settings
        super().__init__(particulator=particulator)

    def run(self):
        return super()._run(self.settings.nt, self.settings.steps_per_output_interval)
