from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence, Condensation, Displacement, EulerianAdvection, AmbientThermodynamics, Freezing
from PySDM.environments import Kinematic2D
from PySDM.initialisation import spectral_sampling, spatial_sampling, spectro_glacial
from PySDM import products as PySDM_products
from .mpdata_2d import MPDATA_2D
from PySDM_examples.utils import DummyController
import numpy as np


class Simulation:

    def __init__(self, settings, storage, SpinUp, backend=CPU):
        self.settings = settings
        self.storage = storage
        self.particulator = None
        self.backend = backend
        self.SpinUp = SpinUp

    @property
    def products(self):
        return self.particulator.products

    def reinit(self, products=None):
        builder = Builder(n_sd=self.settings.n_sd, backend=self.backend, formulae=self.settings.formulae)
        environment = Kinematic2D(dt=self.settings.dt,
                                  grid=self.settings.grid,
                                  size=self.settings.size,
                                  rhod_of=self.settings.rhod_of_zZ)
        builder.set_environment(environment)

        cloud_range = (self.settings.aerosol_radius_threshold, self.settings.drizzle_radius_threshold)
        if products is not None:
            products = list(products)
        products = products or [
            # Note: consider better radius_bins_edges
            PySDM_products.ParticlesWetSizeSpectrum(radius_bins_edges=self.settings.r_bins_edges, normalise_by_dv=True),
            PySDM_products.ParticlesDrySizeSpectrum(radius_bins_edges=self.settings.r_bins_edges, normalise_by_dv=True),
            PySDM_products.TotalParticleConcentration(),
            PySDM_products.TotalParticleSpecificConcentration(),
            PySDM_products.AerosolConcentration(radius_threshold=self.settings.aerosol_radius_threshold),
            PySDM_products.CloudDropletConcentration(radius_range=cloud_range),
            PySDM_products.WaterMixingRatio(name='qc', description_prefix='Cloud', radius_range=cloud_range),
            PySDM_products.WaterMixingRatio(name='qr', description_prefix='Rain', radius_range=(self.settings.drizzle_radius_threshold, np.inf)),
            PySDM_products.DrizzleConcentration(radius_threshold=self.settings.drizzle_radius_threshold),
            PySDM_products.AerosolSpecificConcentration(radius_threshold=self.settings.aerosol_radius_threshold),
            PySDM_products.ParticleMeanRadius(),
            PySDM_products.SuperDropletCount(),
            PySDM_products.RelativeHumidity(), PySDM_products.Pressure(), PySDM_products.Temperature(),
            PySDM_products.WaterVapourMixingRatio(),
            PySDM_products.DryAirDensity(),
            PySDM_products.DryAirPotentialTemperature(),
            PySDM_products.CPUTime(),
            PySDM_products.WallTime(),
            PySDM_products.CloudDropletEffectiveRadius(radius_range=cloud_range)
        ]

        if self.settings.processes['fluid advection']:
            builder.add_dynamic(AmbientThermodynamics())
        if self.settings.processes["condensation"]:
            condensation = Condensation(
                rtol_x=self.settings.condensation_rtol_x,
                rtol_thd=self.settings.condensation_rtol_thd,
                adaptive=self.settings.condensation_adaptive,
                substeps=self.settings.condensation_substeps,
                dt_cond_range=self.settings.condensation_dt_cond_range,
                schedule=self.settings.condensation_schedule
            )
            builder.add_dynamic(condensation)
            products.append(PySDM_products.CondensationTimestepMin())
            products.append(PySDM_products.CondensationTimestepMax())
            products.append(PySDM_products.PeakSupersaturation())
        displacement = None
        if self.settings.processes["particle advection"]:
            displacement = Displacement(enable_sedimentation=self.settings.processes["sedimentation"])
        if self.settings.processes['fluid advection']:
            initial_profiles = {
                    'th': self.settings.initial_dry_potential_temperature_profile,
                    'qv': self.settings.initial_vapour_mixing_ratio_profile
                }
            advectees = dict(
                (key, np.repeat(
                    profile.reshape(1, -1),
                    environment.mesh.grid[0],
                    axis=0)
                 ) for key, profile in initial_profiles.items()
            )
            solver = MPDATA_2D(
                advectees=advectees,
                stream_function=self.settings.stream_function,
                rhod_of_zZ=self.settings.rhod_of_zZ,
                dt=self.settings.dt,
                grid=self.settings.grid,
                size=self.settings.size,
                displacement=displacement,
                n_iters=self.settings.mpdata_iters,
                infinite_gauge=self.settings.mpdata_iga,
                flux_corrected_transport=self.settings.mpdata_fct,
                third_order_terms=self.settings.mpdata_tot
            )
            builder.add_dynamic(EulerianAdvection(solver))
        if self.settings.processes["particle advection"]:
            builder.add_dynamic(displacement)
            products.append(PySDM_products.SurfacePrecipitation())
        if self.settings.processes["coalescence"]:
            builder.add_dynamic(Coalescence(
                kernel=self.settings.kernel,
                adaptive=self.settings.coalescence_adaptive,
                dt_coal_range=self.settings.coalescence_dt_coal_range,
                substeps=self.settings.coalescence_substeps,
                optimized_random=self.settings.coalescence_optimized_random
            ))
            products.append(PySDM_products.CoalescenceTimestepMean())
            products.append(PySDM_products.CoalescenceTimestepMin())
            products.append(PySDM_products.CollisionRate())
            products.append(PySDM_products.CollisionRateDeficit())
            products.append(PySDM_products.ActivatingRate())
            products.append(PySDM_products.DeactivatingRate())
            products.append(PySDM_products.RipeningRate())
        if self.settings.processes["freezing"]:
            builder.add_dynamic(Freezing())
            products.append(PySDM_products.IceWaterContent())
            products.append(PySDM_products.FreezableSpecificConcentration(self.settings.T_bins_edges))
            products.append(PySDM_products.ParticlesConcentration(specific=True))
        if self.settings.processes["PartMC piggy-backer"]:
            products.append(PySDM_products.PartMC.VolumeFractalDimension())

        kw = {}
        if self.settings.processes["freezing"]:
            kw['spectro_glacial_discretisation'] = spectro_glacial.Independent(
                size_spectrum=self.settings.spectrum_per_mass_of_dry_air,
                freezing_temperature_spectrum=self.settings.formulae.freezing_temperature_spectrum
            )
        else:
            kw['spectral_discretisation'] = spectral_sampling.UniformRandom(
                spectrum=self.settings.spectrum_per_mass_of_dry_air
            )

        attributes = environment.init_attributes(
            spatial_discretisation=spatial_sampling.Pseudorandom(),
            **kw,
            kappa=self.settings.kappa
        )

        if self.settings.processes["freezing"]:
            attributes['spheroid mass'] = np.zeros(self.settings.n_sd),

        self.particulator = builder.build(attributes, products)
        if self.SpinUp is not None:
            self.SpinUp(self.particulator, self.settings.n_spin_up)
        if self.storage is not None:
            self.storage.init(self.settings)

    def run(self, controller=DummyController(), vtk_exporter=None):
        with controller:
            for step in self.settings.output_steps:
                if controller.panic:
                    break

                self.particulator.run(step - self.particulator.n_steps)

                self.store(step)
                
                if vtk_exporter is not None:
                    vtk_exporter.export_particles(self.particulator)

                controller.set_percent(step / self.settings.output_steps[-1])

    def store(self, step):
        for name, product in self.particulator.products.items():
            self.storage.save(product.get(), step, name)
