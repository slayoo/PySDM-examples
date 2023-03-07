from PySDM.physics import si


class SettingsCoalescence:
    def __init__(self):
        self.rho = 1 * si.kg / si.m**3
        self.total_number_0 = 1e6
        self.drop_mass_0 = 0.001 * si.kg
        self.total_volume = self.total_number_0 * self.drop_mass_0 / self.rho
        self.c = 0.0001 / si.s
        self.dt = 1 * si.s
        self.dv = 1 * si.m**3
        self.frag_mass = -1 * si.kg  # should not matter here at all!
        self.prods = ("total volume", "total number", "super-particle count")

        self.n_sds = [2**power for power in range(3, 11, 2)]


class SettingsBreakupCoalescence:
    def __init__(self):
        self.rho = 1 * si.kg / si.m**3
        self.total_number_0 = 1e6
        self.drop_mass_0 = 0.001 * si.kg
        self.total_volume = self.total_number_0 * self.drop_mass_0 / self.rho
        self.c = 0.000001 / si.s
        self.dt = 1 * si.s
        self.dv = 1 * si.m**3

        self.prods = ("total volume", "total number", "super-particle count")

        self.n_sds = [2**power for power in range(3, 11, 2)]

        self.beta = 0.5
        self.frag_mass = 0.1 * si.g
