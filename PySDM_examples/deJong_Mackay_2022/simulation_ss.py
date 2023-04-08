import pickle as pkl

import numpy as np
from PySDM.dynamics.collisions.breakup_fragmentations import (
    AlwaysN,
    Gaussian,
    Straub2010Nf,
)
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc, Straub2010Ec
from PySDM.initialisation.spectra import Exponential
from PySDM.physics import si
from PySDM.physics.constants import si

from PySDM_examples.deJong_Mackay_2022 import Settings0D, run_box_breakup


def run_to_steady_state(parameterization, n_sd, steps, nruns=1, dt=1 * si.s):
    rain_rate = 54 * si.mm / si.h
    mp_scale = 4.1e3 * (rain_rate / si.mm * si.h) ** (-0.21) / si.m
    mp_N0 = 8e6 / si.m**4
    n_part = mp_N0 / mp_scale

    nbins = 81

    y_ensemble = np.zeros((nruns, len(steps), nbins - 1))
    irun = 0

    while irun < nruns:
        if parameterization == "Straub2010":
            settings = Settings0D(
                seed=7 ** (irun + 1),
                fragmentation=Straub2010Nf(
                    vmin=(0.05 * si.mm) ** 3 * np.pi / 6, nfmax=10000
                ),
            )
            settings.coal_eff = Straub2010Ec()
            # settings.coal_eff = ConstEc(Ec=0.2)
        else:
            print("parameterization not recognized")
            return

        settings.dv = 1e6 * si.m**3
        settings.dt = dt
        settings.spectrum = Exponential(
            norm_factor=n_part * settings.dv, scale=1 / mp_scale
        )
        settings.n_sd = n_sd
        settings.radius_bins_edges = np.linspace(
            0 * si.mm, 2 * si.mm, num=nbins, endpoint=True
        )
        dr = np.diff(settings.radius_bins_edges) / si.mm

        settings.warn_overflows = False
        settings._steps = steps  # pylint: disable=protected-access
        try:
            (x, y, rates) = run_box_breakup(
                settings, sample_in_radius=True, return_nv=True
            )
            y_ensemble[irun] = y
            print("Success with run #" + str(irun + 1))
            irun += 1
        except:
            if dt > 0.5 * si.s:
                print(
                    "Error in steady state sim for "
                    + str(n_sd)
                    + " superdroplets, moving on with dt="
                    + str(dt / 2)
                )
                dt = dt / 2
            else:
                print(
                    "Error in steady state sim for "
                    + str(n_sd)
                    + " superdroplets, proceeding to next iteration"
                )
                rates = np.nan
                x = (settings.radius_bins_edges[:-1] / si.micrometres,)[0]
                y_ensemble[irun] = np.ones((len(steps), nbins - 1)) * np.nan
                irun += 1
                dt = 1 * si.s

    data_filename = "data/steadystate_" + parameterization + "_" + str(n_sd) + "sd.pkl"

    with open(data_filename, "wb") as handle:
        pkl.dump((x, y_ensemble, rates), handle, protocol=pkl.HIGHEST_PROTOCOL)


def get_straub_fig10_init():
    rain_rate = 54 * si.mm / si.h
    mp_scale = 4.1e3 * (rain_rate / si.mm * si.h) ** (-0.21) / si.m
    mp_N0 = 8e6 / si.m**4

    straub_x_init = np.linspace(0.0, 4.0, 100) * si.mm
    straub_y_init = mp_N0 * np.exp(-1.0 * mp_scale * (straub_x_init)) * si.mm

    return (straub_x_init, straub_y_init)


def get_straub_fig10_data():
    graph_x = np.array(
        [
            0.08988764,
            0.086142322,
            0.097378277,
            0.108614232,
            0.119850187,
            0.142322097,
            0.164794007,
            0.194756554,
            0.224719101,
            0.262172285,
            0.314606742,
            0.36329588,
            0.419475655,
            0.479400749,
            0.558052434,
            0.68164794,
            0.816479401,
            0.943820225,
            1.071161049,
            1.213483146,
            1.370786517,
            1.617977528,
            1.865168539,
            2.074906367,
            2.322097378,
            2.546816479,
            2.801498127,
            3.018726592,
            3.220973783,
            3.378277154,
            3.543071161,
            3.651685393,
        ]
    )
    graph_log_y = np.array(
        [
            1.055803251,
            1.003199334,
            1.14357294,
            1.316140369,
            1.509917836,
            1.811447329,
            2.159352957,
            2.607176908,
            3.048912888,
            3.453402358,
            3.849578422,
            3.955529874,
            3.849578422,
            3.634485506,
            3.328848104,
            2.897005075,
            2.525213782,
            2.294463222,
            2.125139584,
            2.045222871,
            2.02571776,
            2.032198707,
            2,
            1.930947254,
            1.811447329,
            1.685826681,
            1.531778159,
            1.389585281,
            1.262606891,
            1.169430766,
            1.069379699,
            1.0064089029687935,
        ]
    )
    return (graph_x, graph_log_y)
