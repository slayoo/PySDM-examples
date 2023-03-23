import pickle as pkl

import numpy as np
from PySDM.dynamics.collisions.breakup_fragmentations import (
    AlwaysN,
    Gaussian,
    LowList1982Nf,
    Straub2010Nf,
)
from PySDM.dynamics.collisions.coalescence_efficiencies import (
    ConstEc,
    LowList1982Ec,
    Straub2010Ec,
)
from PySDM.initialisation.spectra import Exponential
from PySDM.physics.constants import si

from PySDM_examples.deJong_Mackay_2022 import Settings0D, run_box_breakup


def run_to_steady_state(parameterization, n_sd, steps, nruns=1):
    rain_rate = 54 * si.mm / si.h
    mp_scale = 4.1e3 * (rain_rate / si.mm * si.h) ** (-0.21) / si.m
    mp_N0 = 8e6 / si.m**4
    n_part = mp_N0 / mp_scale

    nbins = 81

    y_ensemble = np.zeros((nruns, len(steps), nbins - 1))

    for irun in range(nruns):
        if parameterization == "Straub2010":
            settings = Settings0D(
                seed=7 ** (irun + 1),
                fragmentation=Straub2010Nf(
                    vmin=(0.1 * si.mm) ** 3 * np.pi / 6, nfmax=1000
                ),
            )
            settings.coal_eff = Straub2010Ec()
        else:
            print("parameterization not recognized")
            return

        settings.dv = 1e6 * si.m**3
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

        (x, y, rates) = run_box_breakup(settings, sample_in_radius=True, return_nv=True)
        y_ensemble[irun] = y

    data_filename = "data/steadystate_" + parameterization + "_" + str(n_sd) + "sd.pkl"

    with open(data_filename, "wb") as handle:
        pkl.dump((x, y_ensemble, rates), handle, protocol=pkl.HIGHEST_PROTOCOL)
