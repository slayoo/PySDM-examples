from chempy import Substance
from PySDM.initialisation import spectra
from PySDM.physics import constants_defaults as const
from PySDM.physics import si
from pystrict import strict

compounds = ("(NH4)2SO4", "bcary_dark", "bcary_light", "apinene_dark", "apinene_light")

molar_masses = {
    "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass * si.gram / si.mole,
    "bcary_dark": 299 * si.gram / si.mole,
    "bcary_light": 360 * si.gram / si.mole,
    "apinene_dark": 209 * si.gram / si.mole,
    "apinene_light": 265 * si.gram / si.mole,
}

densities = {
    "(NH4)2SO4": 1.77 * si.g / si.cm ** 3,
    "bcary_dark": 1.20 * si.g / si.cm ** 3,
    "bcary_light": 1.40 * si.g / si.cm ** 3,
    "apinene_dark": 1.27 * si.g / si.cm ** 3,
    "apinene_light": 1.51 * si.g / si.cm ** 3,
}

is_organic = {
    "(NH4)2SO4": False,
    "bcary_dark": True,
    "bcary_light": True,
    "apinene_dark": True,
    "apinene_light": True,
}

ionic_dissociation_phi = {
    "(NH4)2SO4": 3,
    "bcary_dark": 1,
    "bcary_light": 1,
    "apinene_dark": 1,
    "apinene_light": 1,
}


def volume_fractions(mass_fractions: dict):
    volume_fractions = {
        k: (mass_fractions[k] / densities[k])
        / sum(mass_fractions[i] / densities[i] for i in compounds)
        for k in compounds
    }
    return volume_fractions


def f_org_volume(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    return sum(is_organic[k] * volfrac[k] for k in compounds)


def kappa(mass_fractions: dict):
    kappa = {}
    for model in (
        "bulk",
        "CompressedFilmOvadnevaite",
        "CompressedFilmRuehl",
        "SzyszkowskiLangmuir",
    ):
        volfrac = volume_fractions(mass_fractions)
        molar_volumes = {i: molar_masses[i] / densities[i] for i in compounds}

        _masked = {k: (not is_organic[k]) * volfrac[k] for k in compounds}
        volume_fractions_of_just_inorg = {
            k: _masked[k] / sum(list(_masked.values())) for k in compounds
        }

        if model in (
            "CompressedFilmOvadnevaite",
            "CompressedFilmRuehl",
            "SzyszkowskiLangmuir",
        ):
            ns_per_vol = (1 - f_org_volume(mass_fractions)) * sum(
                ionic_dissociation_phi[i]
                * volume_fractions_of_just_inorg[i]
                / molar_volumes[i]
                for i in compounds
            )
        elif model == "bulk":
            ns_per_vol = sum(
                ionic_dissociation_phi[i] * volfrac[i] / molar_volumes[i]
                for i in compounds
            )
        else:
            raise AssertionError()
        kappa[model] = ns_per_vol * const.Mv / const.rho_w

    return kappa


def nu_org(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    molar_volumes = {i: molar_masses[i] / densities[i] for i in compounds}

    _masked = {k: (is_organic[k]) * volfrac[k] for k in compounds}
    volume_fractions_of_just_org = {
        k: _masked[k] / sum(list(_masked.values())) for k in compounds
    }

    _nu = sum(volume_fractions_of_just_org[i] * molar_volumes[i] for i in compounds)
    return _nu


class _Aerosol:
    pass


@strict
class AerosolBetaCaryophylleneDark(_Aerosol):
    def __init__(self, Forg: float = 0.8, N: float = 400):
        mode = {
            "(NH4)2SO4": (1 - Forg),
            "bcary_dark": Forg,
            "bcary_light": 0,
            "apinene_dark": 0,
            "apinene_light": 0
        }
        self.aerosol_modes_per_cc = (
            {
                "f_org": f_org_volume(mode),
                "kappa": kappa(mode),
                "nu_org": nu_org(mode),
                "spectrum": spectra.Lognormal(
                    norm_factor=N / si.cm ** 3, m_mode=50.0 * si.nm, s_geom=1.75
                ),
            },
        )

    color = "red"
    
class AerosolBetaCaryophylleneLight(_Aerosol):
    def __init__(self, Forg: float = 0.8, N: float = 400):
        mode = {
            "(NH4)2SO4": (1 - Forg),
            "bcary_dark": 0,
            "bcary_light": Forg,
            "apinene_dark": 0,
            "apinene_light": 0
        }
        self.aerosol_modes_per_cc = (
            {
                "f_org": f_org_volume(mode),
                "kappa": kappa(mode),
                "nu_org": nu_org(mode),
                "spectrum": spectra.Lognormal(
                    norm_factor=N / si.cm ** 3, m_mode=50.0 * si.nm, s_geom=1.75
                ),
            },
        )

    color = "orange"

@strict
class AerosolAlphaPineneDark(_Aerosol):
    def __init__(self, Forg: float = 0.8, N: float = 400):
        mode = {
            "(NH4)2SO4": (1 - Forg),
            "bcary_dark": 0,
            "bcary_light": 0,
            "apinene_dark": Forg,
            "apinene_light": 0
        }
        self.aerosol_modes_per_cc = (
            {
                "f_org": f_org_volume(mode),
                "kappa": kappa(mode),
                "nu_org": nu_org(mode),
                "spectrum": spectra.Lognormal(
                    norm_factor=N / si.cm ** 3, m_mode=50.0 * si.nm, s_geom=1.75
                ),
            },
        )

    color = "green"
    
@strict
class AerosolAlphaPineneLight(_Aerosol):
    def __init__(self, Forg: float = 0.8, N: float = 400):
        mode = {
            "(NH4)2SO4": (1 - Forg),
            "bcary_dark": 0,
            "bcary_light": 0,
            "apinene_dark": 0,
            "apinene_light": Forg
        }
        self.aerosol_modes_per_cc = (
            {
                "f_org": f_org_volume(mode),
                "kappa": kappa(mode),
                "nu_org": nu_org(mode),
                "spectrum": spectra.Lognormal(
                    norm_factor=N / si.cm ** 3, m_mode=50.0 * si.nm, s_geom=1.75
                ),
            },
        )

    color = "green"
