import numpy as np
from matplotlib import pyplot
from PySDM.dynamics import Coalescence, Collision
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import ConstantSize
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision_kernels import ConstantK
from PySDM.physics import si

from .equations import Equations, EquationsHelpers
from .settings import Settings
from .simulation import Simulation

# hardcode seed[s] (i.e., to be able to reproduce paper plots)

# different initial conditions (currently flat with const multiplicity, worth checking )

# smoke tests with asserts
# GPU
# unit test na frag_size > drop_size_0
# all rates should be in 1/dv units !
# unit tests for same effect of changing dt or c

NO_BOUNCE = ConstEb(1)


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
    # coalescence + breakup analytic
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


def coalescence_eq10(
    settings=None, n_steps=256, n_realisations=2, title=None, plot=False
):
    # arrange
    seeds = [i for i in range(n_realisations)]
    settings = settings or Settings(
        srivastava_c=0.0001 / si.s, n_sds=[2**power for power in range(3, 11, 2)]
    )

    simulation = Simulation(
        n_steps=n_steps,
        settings=settings,
        collision_dynamic=Coalescence(collision_kernel=ConstantK(a=settings.c)),
    )

    x = np.arange(n_steps + 1, dtype=float)

    equations = Equations(
        M=settings.total_volume * settings.rho / settings.frag_mass, c=settings.c
    )
    equation_helper = EquationsHelpers(
        settings.total_volume,
        settings.total_number_0,
        settings.rho,
        frag_mass=settings.frag_mass,
    )
    m0 = equation_helper.m0()

    x_log = compute_log_space(x)
    analytic_results = {
        "coal": get_coalescence_analytic_results(equations, settings, m0, x, x_log),
    }

    # act
    sim_products = simulation.run(x, seeds=seeds)
    secondary_products = get_pysdm_secondary_products(
        products=sim_products, total_volume=settings.total_volume
    )

    pysdm_results = get_processed_results(secondary_products)

    plot_prods = [
        k for k in list(pysdm_results.values())[0].keys() if k != "total volume"
    ]

    # plot
    add_to_plot_simulation_results(
        plot_prods,
        settings.n_sds,
        x,
        pysdm_results,
        analytic_results,
        analytic_keys=analytic_results.keys(),
        title=title,
    )
    if plot:
        pyplot.show()

    # assert
    for assert_prod in ("mean drop volume / total volume %",):
        np.testing.assert_allclose(
            actual=pysdm_results[settings.n_sds[-1]][assert_prod]["avg"],
            desired=analytic_results["coal"][assert_prod],
            rtol=2e-1,
        )

    return pysdm_results, analytic_results


# note: for breakup to be representable:
#       - frag_mass must be smaller than drop size
#       - breakup rate beta and the multiplicities must make sense :)
#         (i.e., no total breakup in first timestep)


def test_coalescence_and_breakup_cases():
    cases = (
        ("merging only", 0.5e-6 / si.s, 1e-15 / si.s, -1 * si.g),
        ("breakup only", 1e-15 / si.s, 1e-9 / si.s, 0.25 * si.g),
        ("merge + break", 0.5e-6 / si.s, 1e-9 / si.s, 0.25 * si.g),
    )
    for title, c, beta, frag_mass in cases:
        settings = Settings(
            srivastava_c=c,
            srivastava_beta=beta,
            frag_mass=frag_mass,
            drop_mass_0=1 * si.g,
            dt=1 * si.s,
            n_sds=[2**power for power in range(3, 14, 3)],
        )
        coalescence_and_breakup_eq13(settings, title=title)


def coalescence_and_breakup_eq13(
    settings=None, n_steps=256, n_realisations=2, title=None, plot=False
):
    # arrange
    seeds = [i for i in range(n_realisations)]

    # c in analytics is this settings.c or settings.c / collision_rate
    collision_rate = (
        settings.c + settings.beta
    )  # TODO !!! there is no such notion in the paper
    simulation = Simulation(
        n_steps=n_steps,
        settings=settings,
        collision_dynamic=Collision(
            collision_kernel=ConstantK(a=collision_rate),
            coalescence_efficiency=ConstEc(settings.c / collision_rate),
            breakup_efficiency=NO_BOUNCE,
            fragmentation_function=ConstantSize(c=settings.frag_mass / settings.rho),
        ),
    )

    x = np.arange(n_steps + 1, dtype=float)
    sim_products = simulation.run(x, seeds=seeds)

    secondary_products = get_pysdm_secondary_products(
        products=sim_products, total_volume=settings.total_volume
    )

    pysdm_results = get_processed_results(secondary_products)

    equations = Equations(
        M=settings.total_volume * settings.rho / settings.frag_mass,
        c=settings.c,
        beta=settings.beta,
    )
    equation_helper = EquationsHelpers(
        settings.total_volume,
        settings.total_number_0,
        settings.rho,
        frag_mass=settings.frag_mass,
    )
    m0 = equation_helper.m0()

    x_log = compute_log_space(x)
    analytic_results = {
        # "coal": get_coalescence_analytic_results(equations, settings, m0, x, x_log),
        "": get_breakup_coalescence_analytic_results(equations, settings, m0, x, x_log),
    }

    prods = [k for k in list(pysdm_results.values())[0].keys() if k != "total volume"]

    add_to_plot_simulation_results(
        prods,
        settings.n_sds,
        x,
        pysdm_results=pysdm_results,
        analytic_results=analytic_results,
        analytic_keys=analytic_results.keys(),
        title=f"{title}: frag mass: {settings.frag_mass}kg, c: {settings.c}/s, beta: {settings.beta}/s",
    )

    if plot:
        pyplot.show()

    return pysdm_results, analytic_results


def get_processed_results(res):
    processed = {}
    for n_sd in res.keys():
        processed[n_sd] = {}
        for prod in res[n_sd].keys():
            processed[n_sd][prod] = {}
            all_runs = np.asarray(list(res[n_sd][prod].values()))

            processed[n_sd][prod]["avg"] = np.mean(all_runs, axis=0)
            processed[n_sd][prod]["max"] = np.max(all_runs, axis=0)
            processed[n_sd][prod]["min"] = np.min(all_runs, axis=0)
            processed[n_sd][prod]["std"] = np.std(all_runs, axis=0)

    return processed


# TODO: not needed
def get_pysdm_secondary_products(products, total_volume):
    pysdm_results = products
    for n_sd in products.keys():
        pysdm_results[n_sd]["mean drop volume / total volume %"] = {}

        for k in pysdm_results[n_sd]["total number"].keys():
            pysdm_results[n_sd]["mean drop volume / total volume %"][
                k
            ] = compute_drop_volume_total_volume_ratio(
                mean_volume=total_volume / products[n_sd]["total number"][k],
                total_volume=total_volume,
            )
    return pysdm_results


def get_coalescence_analytic_results(equations, settings, m0, x, x_log):
    mean_mass10 = (
        equations.eq10(m0, equations.tau(x * settings.dt)) * settings.frag_mass
    )
    mean_mass_ratio_log = equations.eq10(m0, equations.tau(x_log * settings.dt))

    return get_analytic_results(equations, settings, mean_mass10, mean_mass_ratio_log)


def get_breakup_coalescence_analytic_results(equations, settings, m0, x, x_log):
    mean_mass13 = (
        equations.eq13(m0, equations.tau(x * settings.dt)) * settings.frag_mass
    )
    mean_mass_ratio_log = equations.eq13(m0, equations.tau(x_log * settings.dt))

    return get_analytic_results(equations, settings, mean_mass13, mean_mass_ratio_log)


def get_analytic_results(equations, settings, mean_mass, mean_mass_ratio):
    res = {}
    res["mean drop volume / total volume %"] = compute_drop_volume_total_volume_ratio(
        mean_volume=mean_mass / settings.rho, total_volume=settings.total_volume
    )
    res["total number"] = equations.M / mean_mass_ratio
    return res


def compute_log_space(x, shift=0, num_points=1000, eps=1e-1):
    assert eps < x[1]
    return (
        np.logspace(np.log10(x[0] if x[0] != 0 else eps), np.log10(x[-1]), num_points)
        + shift
    )


# TODO: not needed
def compute_drop_volume_total_volume_ratio(mean_volume, total_volume):
    return mean_volume / total_volume * 100


def add_to_plot_simulation_results(
    prods,
    n_sds,
    x,
    pysdm_results=None,
    analytic_results=None,
    analytic_keys=None,
    title=None,
):
    # pyplot.style.use("grayscale")

    fig = pyplot.figure(layout="constrained", figsize=(10, 4))
    _wide = 14
    _shrt = 8
    _mrgn = 1
    gs = fig.add_gridspec(nrows=20, ncols=2 * _wide + _shrt + 2 * _mrgn)
    axs = (
        fig.add_subplot(gs[2:, _shrt + _mrgn : _shrt + _mrgn + _wide]),
        fig.add_subplot(gs[2:, 0:_shrt]),
        fig.add_subplot(gs[2:, _shrt + 2 * _mrgn + _wide :]),
    )
    axs[1].set_ylim([6, 3500])

    expons = [3, 5, 7, 9, 11]

    axs[1].set_yscale("log")
    axs[1].set_yticks([2**e for e in expons], [f"$2^{{{e}}}$" for e in expons])

    if title:
        fig.suptitle(title)

    ylims = {}
    for prod in prods:
        ylims[prod] = 0
        for n_sd in n_sds:
            ylims[prod] = max(ylims[prod], np.amax(pysdm_results[n_sd][prod]["max"]))

    for i, prod in enumerate(prods):
        # plot numeric
        if pysdm_results:
            for n_sd in n_sds:  # TODO reversed(n_sds):
                y_model = pysdm_results[n_sd][prod]

                axs[i].step(
                    x,
                    y_model["avg"],
                    where="mid",
                    label=f"initial #SD: {n_sd}",
                    linewidth=1 + np.log(n_sd) / 3,
                    # color=f'#8888{int(np.log(n_sd-4)*13)}'
                )
                axs[i].fill_between(
                    x,
                    y_model["avg"] - y_model["std"],
                    y_model["avg"] + y_model["std"],
                    alpha=0.2,
                )

        # plot analytic
        if analytic_results:
            if analytic_keys:
                for key in analytic_keys:
                    add_analytic_result_to_axs(
                        axs[i],
                        prod,
                        x,
                        analytic_results[key],
                        key=key,
                        ylim=ylims[prod],
                    )
            else:
                add_analytic_result_to_axs(axs[i], prod, x, analytic_results)

        # cosmetics
        axs[i].set_ylabel(prod)

        axs[i].grid()
        axs[i].set_xlabel("step: t / dt")

    axs[0].legend()
    return fig, axs


def add_analytic_result_to_axs(axs_i, prod, x, res, key="", ylim=None):
    if prod != "super-particle count":
        x_theory = x
        y_theory = res[prod]

        if prod == "total number":
            if y_theory.shape != x_theory.shape:
                x_theory = compute_log_space(x)

                axs_i.set_yscale("log")
                axs_i.set_xscale("log")
                axs_i.set_xlim(x_theory[0], None)

        if prod == "total volume":
            axs_i.set_ylim(0, 1.25 * ylim)

        axs_i.plot(
            x_theory,
            y_theory,
            label=f"analytic {key}",
            linestyle=":",
            linewidth=2,
            color="black",
        )
