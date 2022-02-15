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
        n_sd = settings.n_sd_per_mode * len(settings.aerosol.aerosol_modes_per_cc)
        builder = Builder(n_sd=n_sd, backend=CPU(formulae=settings.formulae))
        builder.set_environment(env)

        attributes = {
            'dry volume':np.empty(0),
            'dry volume organic':np.empty(0),
            'kappa times dry volume':np.empty(0),
            'n': np.ndarray(0)
        }
        for mode in settings.aerosol.aerosol_modes_per_cc:
            r_dry, N = settings.spectral_sampling(
                spectrum=mode['spectrum']).sample(settings.n_sd_per_mode)
            dv = (settings.rho0 / settings.mass_of_dry_air)
            n_in_dv = N / dv
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            attributes['n'] = np.append(attributes['n'], n_in_dv)
            attributes['dry volume'] = np.append(attributes['dry volume'], v_dry)
            attributes['dry volume organic'] = np.append(
                attributes['dry volume organic'], mode['f_org'] * v_dry)
            attributes['kappa times dry volume'] = np.append(
                attributes['kappa times dry volume'], v_dry * mode['kappa'])
        for attribute in attributes.values():
            assert attribute.shape[0] == n_sd

        np.testing.assert_approx_equal(
            np.sum(attributes['n']) * dv,
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
                radius_bins_edges=settings.wet_radius_bins_edges),
        )

        self.particulator = builder.build(attributes=attributes, products=products)
        self.settings = settings

    def run(self):
        return super()._run(self.settings.nt, self.settings.steps_per_output_interval)
