from pystrict import strict
from chempy import Substance
from PySDM.initialisation import spectra
from PySDM.physics import si
from PySDM.physics.constants_defaults import rho_w, Mv

compounds = ('(NH4)2SO4', 'NaCl')

molar_masses = {
    "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass * si.gram / si.mole,
    "NaCl": Substance.from_formula("NaCl").mass * si.gram / si.mole,
}

densities = {
    '(NH4)2SO4': 1.77 * si.g / si.cm**3,
    'NaCl': 2.16 * si.g / si.cm**3
}

is_organic = {
    '(NH4)2SO4': False,
    'NaCl': False
}

ionic_dissociation_phi = {
    '(NH4)2SO4': 3,
    'NaCl': 2
}


def volume_fractions(mass_fractions: dict):
    return {
        k: (mass_fractions[k] / densities[k]) / sum(
            mass_fractions[i] / densities[i] for i in compounds
        ) for k in compounds
    }


def f_org_volume(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    return sum(is_organic[k] * volfrac[k] for k in compounds)


def kappa(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    molar_volumes = {i: molar_masses[i] / densities[i] for i in compounds}
    ns_per_vol = sum(ionic_dissociation_phi[i] * volfrac[i] / molar_volumes[i]
                        for i in compounds)
    result = ns_per_vol * Mv / rho_w

    return result


def nu_org(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    molar_volumes = {i: molar_masses[i] / densities[i] for i in compounds}

    _masked = {k: (is_organic[k]) * volfrac[k] for k in compounds}
    _tot_org = sum(list(_masked.values()))
    if _tot_org == 0:
        volume_fractions_of_just_org = {k:0.0 for k in compounds}
    else:
        volume_fractions_of_just_org = {k:_masked[k] / _tot_org for k in compounds}

    _nu = sum(volume_fractions_of_just_org[i] * molar_volumes[i] for i in compounds)
    return _nu


class Aerosol:
    pass


@strict
class AerosolMarine(Aerosol):
    Aitken = {'(NH4)2SO4': 1.0, 'NaCl': 0}
    Accumulation = {'(NH4)2SO4': 0, 'NaCl': 1.0}

    def __init__(self):
        self.aerosol_modes_per_cc = (
        {
            'f_org': f_org_volume(self.Aitken),
            'kappa': kappa(self.Aitken),
            'nu_org': nu_org(self.Aitken),
            'spectrum': spectra.Lognormal(
                norm_factor=850 / si.cm ** 3,
                m_mode=15 * si.nm,
                s_geom=1.6
            )
        },
        {
            'f_org': f_org_volume(self.Accumulation),
            'kappa': kappa(self.Accumulation),
            'nu_org': nu_org(self.Accumulation),
            'spectrum': spectra.Lognormal(
                norm_factor=10 / si.cm ** 3,
                m_mode=850 * si.nm,
                s_geom=1.2
            ),
        }
    )
    color = 'dodgerblue'
