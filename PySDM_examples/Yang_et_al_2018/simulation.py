from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import AmbientThermodynamics
from PySDM.dynamics import Condensation
from PySDM.environments import Parcel
from PySDM.physics.formulae import Formulae
import PySDM.products as PySDM_products


class Simulation:

    def __init__(self, settings, backend=CPU):
        dt_output = settings.total_time / settings.n_steps  # TODO #334 overwritten in jupyter example
        self.n_substeps = 1  # TODO #334 use condensation substeps
        while (dt_output / self.n_substeps >= settings.dt_max):
            self.n_substeps += 1
        self.formulae = Formulae(condensation_coordinate=settings.coord, saturation_vapour_pressure='AugustRocheMagnus')
        self.bins_edges = self.formulae.trivia.volume(settings.r_bins_edges)
        builder = Builder(backend=backend(formulae=self.formulae), n_sd=settings.n_sd)
        builder.set_environment(Parcel(
            dt=dt_output / self.n_substeps,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.p0,
            q0=settings.q0,
            T0=settings.T0,
            w=settings.w,
            z0=settings.z0
        ))

        environment = builder.particulator.environment
        builder.add_dynamic(AmbientThermodynamics())
        condensation = Condensation(
            adaptive=settings.adaptive,
            rtol_x=settings.rtol_x,
            rtol_thd=settings.rtol_thd,
            dt_cond_range=settings.dt_cond_range
        )
        builder.add_dynamic(condensation)

        products = [
            PySDM_products.ParticlesWetSizeSpectrum(radius_bins_edges=settings.r_bins_edges),
            PySDM_products.CondensationTimestepMin(),
            PySDM_products.CondensationTimestepMax(),
            PySDM_products.RipeningRate()
        ]

        attributes = environment.init_attributes(
            n_in_dv=settings.n,
            kappa=settings.kappa,
            r_dry=settings.r_dry
        )

        self.particulator = builder.build(attributes, products)

        self.n_steps = settings.n_steps

    def save(self, output):
        cell_id = 0
        output["r_bins_values"].append(self.particulator.products["Particles Wet Size Spectrum"].get())
        volume = self.particulator.attributes['volume'].to_ndarray()
        output["r"].append(self.formulae.trivia.radius(volume=volume))
        output["S"].append(self.particulator.environment["RH"][cell_id] - 1)
        output["qv"].append(self.particulator.environment["qv"][cell_id])
        output["T"].append(self.particulator.environment["T"][cell_id])
        output["z"].append(self.particulator.environment["z"][cell_id])
        output["t"].append(self.particulator.environment["t"][cell_id])
        output["dt_cond_max"].append(self.particulator.products["dt_cond_max"].get()[cell_id].copy())
        output["dt_cond_min"].append(self.particulator.products["dt_cond_min"].get()[cell_id].copy())
        output['ripening_rate'].append(self.particulator.products['ripening_rate'].get()[cell_id].copy())

    def run(self):
        output = {"r": [], "S": [], "z": [], "t": [], "qv": [], "T": [],
                  "r_bins_values": [], "dt_cond_max": [], "dt_cond_min": [], "ripening_rate": []}

        self.save(output)
        for step in range(self.n_steps):
            self.particulator.run(self.n_substeps)
            self.save(output)
        return output
