from PySDM.physics import si

DUMMY_FRAG_MASS = -1


class SimProducts:
    total_numer = "total numer"


class Settings:
    """interprets parameters from Srivastava 1982 in PySDM context"""

    def __init__(
        self,
        *,
        n_sds,
        srivastava_c,
        srivastava_beta=None,
        frag_mass=DUMMY_FRAG_MASS,
        # TODO: get rid of the surplus parameters below
        dt=1 * si.s,
        dv=1 * si.m**3,
        drop_mass_0=1 * si.g,
        rho=1 * si.kg / si.m**3,
        total_number=1e6
    ):
        self.rho = rho
        self.total_number_0 = total_number
        self.drop_mass_0 = drop_mass_0
        self.total_volume = self.total_number_0 * self.drop_mass_0 / self.rho
        self.dt = dt
        self.dv = dv
        self.frag_mass = frag_mass

        self.prods = ("total volume", "total number", "super-particle count")
        self.n_sds = n_sds

        # TODO
        self.c = srivastava_c
        self.beta = srivastava_beta
