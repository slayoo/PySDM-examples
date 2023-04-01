from PySDM.physics import si

DUMMY_FRAG_MASS = -1


class SimProd:
    def __init__(self, name, plot_title=None, plot_xscale=None, plot_yscale=None):
        self.name = name
        self.plot_title = plot_title or name
        self.plot_yscale = plot_yscale
        self.plot_xscale = plot_xscale


class SimProducts:
    class PySDM:
        total_numer = SimProd(
            name="total numer",
            plot_title="total droplet numer",
            plot_xscale="log",
            plot_yscale="log",
        )
        total_volume = SimProd(name="total volume")
        super_particle_count = SimProd(
            name="super-particle count", plot_xscale="log", plot_yscale="log"
        )

    class Computed:
        mean_drop_volume_total_volume_ratio = SimProd(
            name="mean drop volume / total volume %"
        )


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

        self.prods = (
            SimProducts.PySDM.total_volume.name,
            SimProducts.PySDM.total_numer.name,
            SimProducts.PySDM.super_particle_count.name,
        )
        self.n_sds = n_sds

        # TODO
        self.c = srivastava_c
        self.beta = srivastava_beta
