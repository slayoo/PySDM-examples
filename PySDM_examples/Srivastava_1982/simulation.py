import numpy as np
from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.products import SuperDropletCountPerGridbox, VolumeFirstMoment, ZerothMoment


class Simulation:
    def __init__(self, n_steps, settings, collision_dynamic=None):
        self.backend_class = CPU
        self.collision_dynamic = collision_dynamic
        self.settings = settings

        formulae = Formulae(fragmentation_function="ConstantSize")

        self.backend = self.backend_class(formulae)
        self.n_steps = n_steps

        self.simulation_res = {
            n_sd: {prod: np.empty(n_steps + 1) for prod in self.settings.prods}
            for n_sd in self.settings.n_sds
        }

    def run(self, x):
        for n_sd in self.settings.n_sds:
            builder = Builder(backend=self.backend, n_sd=n_sd)
            builder.set_environment(Box(dt=self.settings.dt, dv=self.settings.dv))
            builder.add_dynamic(self.collision_dynamic)
            particulator = builder.build(
                products=(
                    VolumeFirstMoment(name="total volume"),
                    SuperDropletCountPerGridbox(name="super-particle count"),
                    ZerothMoment(name="total number"),
                ),
                attributes={
                    "n": np.full(n_sd, self.settings.total_number_0 / n_sd),
                    "volume": np.full(
                        n_sd, self.settings.total_volume / self.settings.total_number_0
                    ),
                },
            )

            for i in range(len(x)):
                if i != 0:
                    particulator.run(steps=1)
                for prod in self.settings.prods:
                    self.simulation_res[n_sd][prod][i] = particulator.products[
                        prod
                    ].get()
            np.testing.assert_allclose(
                actual=self.simulation_res[n_sd]["total volume"],
                desired=self.settings.total_volume,
                rtol=1e-3,
            )

        return self.simulation_res
