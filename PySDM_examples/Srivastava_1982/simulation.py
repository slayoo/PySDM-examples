import numpy as np
from matplotlib import pyplot
from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.dynamics import Coalescence, Collision
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN, ConstantSize
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

from .equations import Equations, EquationsHelpers

# breakup
# notebooks
# simulation ensembles for different seed but same n_sd
# hardcode seed
# different initial conditions
# smoke tests
# GPU


def test_fig1():
    alpha_star = 1e-5
    beta_star = 1e-4

    pyplot.title("fig 1")
    for m0 in (100, 450):
        eqs = Equations(alpha_star=alpha_star, beta_star=beta_star)
        tau = np.linspace(0, 900)
        y0 = eqs.eq15(m0)
        x = eqs.eq16(tau)
        y = (y0 + np.tanh(x)) / (1 + y0 * np.tanh(x))
        pyplot.plot(tau, eqs.eq15_m_of_y(y), label=f"$m(τ, m_0={m0})$")

    pyplot.axhline(
        eqs.eq12(), linestyle="--", label="$m_E$"
    )  # TODO: value from paper = 215 ???
    pyplot.xlabel("τ")
    pyplot.ylabel("mass")
    pyplot.grid()
    pyplot.legend()
    pyplot.show()


def test_eq_13_14():
    alpha_star = 1e-5
    beta_star = 1e-4
    eqs = Equations(alpha_star=alpha_star, beta_star=beta_star)

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


def test_coalescence():
    backend_class = CPU

    rho = 1 * si.kg / si.m**3
    total_number_0 = 1e6
    drop_mass_0 = 0.001 * si.kg
    total_volume = total_number_0 * drop_mass_0 / rho
    c = 0.0001 / si.s
    dt = 1 * si.s
    dv = 1 * si.m**3
    n_steps = 256
    frag_mass = -1 * si.kg  # should not matter here at all!
    prods = ("total volume", "total number", "super-particle count")

    formulae = Formulae(fragmentation_function="ConstantSize")

    n_sds = [2**power for power in range(3, 11, 2)]

    x = np.arange(n_steps + 1, dtype=float)
    y = {n_sd: {prod: np.empty(n_steps + 1) for prod in prods} for n_sd in n_sds}

    equations = Equations(M=total_volume * rho / frag_mass, c=c)
    equation_helper = EquationsHelpers(
        total_volume, total_number_0, rho, frag_mass=frag_mass
    )
    m0 = equation_helper.m0()
    mean_mass = equations.eq10(m0, equations.tau(x * dt)) * frag_mass

    for n_sd in n_sds:
        builder = Builder(backend=backend_class(formulae), n_sd=n_sd)
        builder.set_environment(Box(dt=dt, dv=dv))
        builder.add_dynamic(Coalescence(collision_kernel=ConstantK(a=c)))
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
                mean_mass_log = equations.eq10(m0, equations.tau(x_theory * dt))
                print(x_theory)
                print(mean_mass_log)
                y_theory = equations.M / mean_mass_log

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
            "mean drop volume / total volume %" if prod == "total volume" else prod
        )
        # axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel("step: t / dt")
    pyplot.savefig("coalescence_only.pdf")
    pyplot.show()


def test_coalescence_and_breakup():
    backend_class = CPU

    rho = 1 * si.kg / si.m**3
    total_number_0 = 1e6
    drop_mass_0 = 0.001 * si.kg
    total_volume = total_number_0 * drop_mass_0 / rho
    c = 0.000001 / si.s
    dt = 1 * si.s
    dv = 1 * si.m**3
    n_steps = 512
    prods = ("total volume", "total number", "super-particle count")

    formulae = Formulae(fragmentation_function="ConstantSize")

    n_sds = [2**power for power in range(3, 11, 2)]

    x = np.arange(n_steps + 1, dtype=float)
    y = {n_sd: {prod: np.empty(n_steps + 1) for prod in prods} for n_sd in n_sds}

    beta = 0.5
    frag_mass = 0.1 * si.g

    equations = Equations(M=total_volume * rho / frag_mass, c=c, beta=beta)
    equation_helper = EquationsHelpers(
        total_volume, total_number_0, rho, frag_mass=frag_mass
    )
    m0 = equation_helper.m0()
    print("test3", m0)

    for n_sd in n_sds:
        builder = Builder(backend=backend_class(formulae), n_sd=n_sd)
        builder.set_environment(Box(dt=dt, dv=dv))
        builder.add_dynamic(
            Collision(
                collision_kernel=ConstantK(a=c),
                coalescence_efficiency=ConstEc(1 - beta),
                breakup_efficiency=ConstEb(beta),
                fragmentation_function=ConstantSize(c=frag_mass / rho),
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

    pysdm_results = y
    for n_sd in y.keys():
        pysdm_results[n_sd]["total volume"] = compute_drop_volume_total_volume_ratio(
            mean_volume=y[n_sd]["total volume"] / y[n_sd]["total number"],
            total_volume=total_volume,
        )

    analytic_results = {"coal": {}, "coal+break": {}}

    mean_mass = equations.eq13(m0, equations.tau(x * dt)) * frag_mass
    analytic_results["coal+break"][
        "total volume"
    ] = compute_drop_volume_total_volume_ratio(
        mean_volume=mean_mass / rho, total_volume=total_volume
    )

    x_log = compute_log_space(x)
    mean_mass_ratio_log = equations.eq13(m0, equations.tau(x_log * dt))
    analytic_results["coal+break"]["total number"] = equations.M / mean_mass_ratio_log

    mean_mass10 = equations.eq10(m0, equations.tau(x * dt)) * frag_mass
    analytic_results["coal"]["total volume"] = compute_drop_volume_total_volume_ratio(
        mean_volume=mean_mass10 / rho, total_volume=total_volume
    )
    print("B", mean_mass10)

    mean_mass_ratio_log = equations.eq10(m0, equations.tau(x_log * dt))
    analytic_results["coal"]["total number"] = equations.M / mean_mass_ratio_log

    plot_simulation_results(
        prods,
        n_sds,
        x,
        pysdm_results,
        analytic_results,
        analytic_keys=analytic_results.keys(),
    )


def compute_log_space(x):
    eps = 1e-2
    return np.logspace(np.log10(x[0] if x[0] != 0 else eps), np.log10(x[-1]), 1000)


def compute_drop_volume_total_volume_ratio(mean_volume, total_volume):
    return mean_volume / total_volume * 100


def plot_simulation_results(
    prods,
    n_sds,
    x,
    pysdm_results=None,
    analytic_results=None,
    analytic_keys=None,
    filename=None,
):
    # pyplot.style.use("grayscale")

    fig, axs = pyplot.subplots(len(prods), 1, figsize=(7, 4 * len(prods)))

    for i, prod in enumerate(prods):
        # plot numeric
        if pysdm_results:
            for n_sd in reversed(n_sds):
                y_model = pysdm_results[n_sd][prod]
                # if prod == "total volume":
                #     y_model = y[n_sd][prod] / y[n_sd]["total number"] / total_volume * 100
                axs[i].step(
                    x,
                    y_model,
                    where="mid",
                    label=f"n_sd = {n_sd}",
                    linewidth=1 + np.log(n_sd) / 3,
                    # color=f'#8888{int(np.log(n_sd-4)*13)}'
                )

        # plot analytic
        if analytic_results:
            if analytic_keys:
                for key in analytic_keys:
                    add_analytic_result_to_axs(
                        axs[i], prod, x, analytic_results[key], key=key
                    )
            else:
                add_analytic_result_to_axs(axs[i], prod, x, analytic_results)

        # cosmetics
        axs[i].set_ylabel(
            "mean drop volume / total volume %" if prod == "total volume" else prod
        )
        # axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel("step: t / dt")

    if filename:
        pyplot.savefig(filename)
    pyplot.show()


def add_analytic_result_to_axs(axs_i, prod, x, res, key=""):
    if prod != "super-particle count":
        x_theory = x
        y_theory = res[prod]

        # if prod == "total volume":
        #     x_theory = x
        #     y_theory = mean_mass / rho / total_volume * 100
        if prod == "total number":
            x_theory = compute_log_space(x)

            axs_i.set_yscale("log")
            axs_i.set_xscale("log")
            axs_i.set_xlim(x_theory[0], None)
        # else:
        #     raise NotImplementedError()
        axs_i.plot(x_theory, y_theory, label=f"analytic {key}", linestyle="-")
