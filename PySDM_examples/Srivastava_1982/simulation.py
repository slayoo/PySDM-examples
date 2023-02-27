import numpy as np
from matplotlib import pyplot

# breakup
# notebooks
# simulation ensembles for different seed but same n_sd
# hardcode seed
# different initial conditions
# smoke tests
# GPU


class Equations:
    def __init__(self, alpha_star, beta_star):
        amb = alpha_star - beta_star
        self._alpha_star = alpha_star
        self._beta_star = beta_star
        self._A = amb / 2 / alpha_star
        self._B = 1 / np.sqrt(
            (0.5 / alpha_star + beta_star / alpha_star)
            + amb**2 / (4 * alpha_star**2)
        )

    def eq12(self):
        return 0.5 + np.sqrt(0.25 + 0.5 / self._alpha_star)

    def eq13(self, m0, tau):
        ebt = np.exp(-self._beta_star * tau)
        return m0 * ebt + (1 + 0.5 / self._beta_star) * (1 - ebt)

    def eq14(self):
        return 1 + 0.5 / self._beta_star

    def eq15(self, m):
        return (m - self._A) * self._B

    def eq15_m_of_y(self, y):
        return (y / self._B) + self._A

    def eq16(self, tau):
        return tau * self._alpha_star / self._B

    @staticmethod
    def eq10(M, dt, c, total_volume, total_number_0, rho, x):
        t = dt * x
        tau = c * M * t
        mean_volume_0 = total_volume / total_number_0
        m0 = rho * mean_volume_0
        mean_mass = m0 + tau / 2
        return mean_mass


def test_simulation():
    alpha_star = 1e-5
    beta_star = 1e-4

    pyplot.title("fig 1")
    for m0 in (100, 450):
        eqs = Equations(alpha_star, beta_star)
        tau = np.linspace(0, 900)
        y0 = eqs.eq15(m0)
        x = eqs.eq16(tau)
        y = (y0 + np.tanh(x)) / (1 + y0 * np.tanh(x))
        pyplot.plot(tau, eqs.eq15_m_of_y(y), label=f"$m(τ, m_0={m0})$")
    pyplot.axhline(eqs.eq12(), linestyle="--", label="$m_E$")
    pyplot.xlabel("τ")
    pyplot.ylabel("mass")
    pyplot.grid()
    pyplot.legend()
    pyplot.show()

    tau = np.linspace(0, 90000)
    m0 = 100
    pyplot.title("equations (13) and (14)")
    pyplot.plot(tau, eqs.eq13(m0, tau), label=f"m(τ, m_0={m0})")
    pyplot.axhline(eqs.eq14(), linestyle="--", label="$m_E$")
    pyplot.xlabel("τ")
    pyplot.ylabel("mass")
    pyplot.grid()
    pyplot.legend()
    pyplot.show()


class ConstantSize:
    pass


def test2():
    from PySDM import Builder, Formulae
    from PySDM.backends import CPU
    from PySDM.dynamics import Collision
    from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
    from PySDM.dynamics.collisions.breakup_fragmentations import ConstantSize
    from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
    from PySDM.dynamics.collisions.collision_kernels import ConstantK
    from PySDM.environments import Box
    from PySDM.physics import si
    from PySDM.products import (
        MeanRadius,
        SuperDropletCountPerGridbox,
        VolumeFirstMoment,
        ZerothMoment,
    )

    backend_class = CPU

    rho = 1 * si.kg / si.m**3
    total_number_0 = 1e6
    frag_size = 1 * si.um**3
    drop_mass_0 = 0.001 * si.kg
    total_volume = total_number_0 * drop_mass_0 / rho
    c = 0.0001  # TODO: unit?
    dt = 1 * si.s
    dv = 1 * si.m**3
    n_steps = 256
    prods = ("total volume", "total number", "super-particle count")

    formulae = Formulae(fragmentation_function="ConstantSize")

    n_sds = [2**power for power in range(3, 11, 2)]

    x = np.arange(n_steps + 1, dtype=float)
    y = {n_sd: {prod: np.empty(n_steps + 1) for prod in prods} for n_sd in n_sds}

    M = total_volume * rho
    mean_mass = Equations.eq10(M, dt, c, total_volume, total_number_0, rho, x)

    for n_sd in n_sds:
        builder = Builder(backend=backend_class(formulae), n_sd=n_sd)
        builder.set_environment(Box(dt=dt, dv=dv))
        builder.add_dynamic(
            Collision(
                collision_kernel=ConstantK(a=c),
                coalescence_efficiency=ConstEc(1),
                breakup_efficiency=ConstEb(0),
                fragmentation_function=ConstantSize(c=frag_size),
            )
        )
        particulator = builder.build(
            products=(
                VolumeFirstMoment(name="total volume"),
                SuperDropletCountPerGridbox(name="super-particle count"),
                ZerothMoment(name="total number"),
            ),
            attributes={
                "n": np.full(n_sd, total_number_0 / n_sd),
                "volume": np.full(n_sd, total_volume / total_number_0),
            },
        )

        print("n_sd", total_volume / n_sd)

        for i in range(len(x)):
            if i != 0:
                particulator.run(steps=1)
            for prod in prods:
                y[n_sd][prod][i] = particulator.products[prod].get()
        np.testing.assert_allclose(
            actual=y[n_sd]["total volume"], desired=total_volume, rtol=1e-3
        )

    pyplot.style.use("grayscale")

    fig, axs = pyplot.subplots(len(prods), 1, figsize=(7, 4 * len(prods)))

    for i, prod in enumerate(prods):
        # plot numeric
        for n_sd in reversed(n_sds):
            y_model = y[n_sd][prod]
            if prod == "total volume":
                y_model = y[n_sd][prod] / y[n_sd]["total number"] / total_volume * 100
            axs[i].step(
                x,
                y_model,
                where="mid",
                label=f"n_sd = {n_sd}",
                linewidth=1 + np.log(n_sd) / 3,
                # color=f'#8888{int(np.log(n_sd-4)*13)}'
            )

        # plot analytic
        if prod != "super-particle count":
            if prod == "total volume":
                x_theory = x
                y_theory = mean_mass / rho / total_volume * 100
            elif prod == "total number":
                eps = 1e-2
                x_theory = np.logspace(
                    np.log10(x[0] if x[0] != 0 else eps), np.log10(x[-1]), 1000
                )
                mean_mass_log = Equations.eq10(
                    M, dt, c, total_volume, total_number_0, rho, x_theory
                )
                print(x_theory)
                print(mean_mass_log)
                y_theory = M / mean_mass_log

                axs[i].set_yscale("log")
                axs[i].set_xscale("log")
                axs[i].set_xlim(eps, None)
            else:
                raise NotImplementedError()
            axs[i].plot(
                x_theory, y_theory, label="analytic", color="red", linestyle="-"
            )

        # cosmetics
        axs[i].set_ylabel(
            "drop volume / total volume %" if prod == "total volume" else prod
        )
        # axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel("step: t / dt")
    pyplot.savefig("coalescence_only.pdf")
    pyplot.show()
