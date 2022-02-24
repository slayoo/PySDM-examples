from pystrict import strict
from PySDM.initialisation import spectra
from PySDM.physics import si
from PySDM.physics import constants_defaults as const
from chempy import Substance

compounds = ('(NH4)2SO4', 'betacary')

molar_masses = {
    "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass * si.gram / si.mole,
    "betacary": 204.36 * si.gram / si.mole
}

densities = {
    '(NH4)2SO4': 1.77 * si.g / si.cm**3,
    'betacary': 0.905 * si.g / si.cm**3
}

is_organic = {
    '(NH4)2SO4': False,
    'betacary': True
}

ionic_dissociation_phi = {
    '(NH4)2SO4': 3,
    'betacary': 1
}

def volume_fractions(mass_fractions: dict):
    volume_fractions = {
        k: (mass_fractions[k] / densities[k]) / sum(
            mass_fractions[i] / densities[i] for i in compounds
        ) for k in compounds
    }
    return volume_fractions

def f_org_volume(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    return sum(is_organic[k] * volfrac[k] for k in compounds)

def kappa(mass_fractions: dict):
    kappa = {}
    for model in ('bulk', 'Ovad', 'Ruehl', 'SL'):
        volfrac = volume_fractions(mass_fractions)
        molar_volumes = {i: molar_masses[i] / densities[i] for i in compounds}

        _masked = {k: (not is_organic[k]) * volfrac[k] for k in compounds}
        volume_fractions_of_just_inorg = {k:_masked[k] / sum(list(_masked.values())) for k in compounds}

        if model == 'Ovad' or 'Ruehl' or 'SL':
            ns_per_vol = (1 - f_org_volume(mass_fractions))  * sum(
                ionic_dissociation_phi[i] * volume_fractions_of_just_inorg[i] / molar_volumes[i] for i in compounds
            )
        elif model == 'bulk':
            ns_per_vol = sum(ionic_dissociation_phi[i] * volfrac[i] / molar_volumes[i] for i in compounds)
        else:
            raise AssertionError()
        kappa[model] = ns_per_vol * const.Mv / const.rho_w

    return kappa

def nu_org(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    molar_volumes = {i: molar_masses[i] / densities[i] for i in compounds}

    _masked = {k: (is_organic[k]) * volfrac[k] for k in compounds}
    volume_fractions_of_just_org = {k:_masked[k] / sum(list(_masked.values())) for k in compounds}
    
    _nu = sum(volume_fractions_of_just_org[i] * molar_volumes[i] for i in compounds)
    return _nu

class _Aerosol:
    pass

@strict
class AerosolBetaCary(_Aerosol):
    def __init__(self, Forg: float = 0.8, N: float = 400):
        mode = {'(NH4)2SO4': (1-Forg), 'betacary': Forg}
        self.aerosol_modes_per_cc = (
            {
                'f_org': f_org_volume(mode),
                'kappa': kappa(mode),
                'nu_org': nu_org(mode),
                'spectrum': spectra.Lognormal(
                    norm_factor = N / si.cm ** 3,
                    m_mode = 50.0 * si.nm,
                    s_geom = 1.75
                )
            },
        )
    color = 'dodgerblue'