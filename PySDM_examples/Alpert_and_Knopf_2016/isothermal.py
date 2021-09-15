from .simulation import simulation
from PySDM.physics import si
from matplotlib import pylab
import matplotlib
from packaging import version
import numpy as np


class Isothermal:
    # note: dv and droplet_volume are dummy multipliers (multiplied and then divided by)
    #       will become used if coalescence or other processes are turned on
    def __init__(self, *, cases, total_time, n_runs_per_case=10, multiplicity=1, dt, droplet_volume=1*si.um**3):
        self.cases = cases
        self.n_runs_per_case = n_runs_per_case
        self.multiplicity = multiplicity
        self.dv = cases.dv
        self.dt = dt
        self.droplet_volume = droplet_volume
        self.total_time = total_time
        self.output = None

    def run(self, keys):
        self.output = {}
        for key in keys:
            case = self.cases[key]
            self.output[key] = []
            for i in range(self.n_runs_per_case):
                number_of_real_droplets = case['ISA'].norm_factor * self.dv
                n_sd = number_of_real_droplets / self.multiplicity
                np.testing.assert_approx_equal(n_sd, int(n_sd))
                n_sd = int(n_sd)

                data = simulation(seed=i, n_sd=n_sd, dt=self.dt, dv=self.dv, spectrum=case['ISA'],
                                  droplet_volume=self.droplet_volume, multiplicity=self.multiplicity, J_het=case['J_het'],
                                  total_time=self.total_time, number_of_real_droplets=number_of_real_droplets)

                self.output[key].append(data)

    def plot(self, ylim, grid=None):
        pylab.rc('font', size=10)
        for key in self.output.keys():
            for run in range(self.n_runs_per_case):
                pylab.step(
                    self.dt / si.min * np.arange(len(self.output[key][run])),
                    self.output[key][run],
                    label=self.cases.label(key) if run == 0 else None,
                    color=self.cases[key]['color'],
                    linewidth=.666
                )
        if version.parse(matplotlib.__version__) >= version.parse('3.3.0'):
            pylab.gca().set_box_aspect(1)
        pylab.legend()
        pylab.yscale('log')
        if grid is not None:
            pylab.grid(which=grid)
        pylab.ylim(ylim)
        pylab.xlim(0, self.total_time / si.min)
        pylab.xlabel("t / min")
        pylab.ylabel("$f_{ufz}$")
