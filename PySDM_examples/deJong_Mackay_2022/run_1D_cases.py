import pickle as pkl

import numpy as np
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import Gaussian
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.physics import si

from PySDM_examples.deJong_Mackay_2022 import Settings1D, Simulation1D

# WITHOUT BREAKUP
n_sd_per_gridbox = 1024
dt = 5 * si.s
dz = 50 * si.m

output = {}
rho_times_w = 3 * si.m / si.s
precip = True
breakup = False
key = f"rhow={rho_times_w}_p={precip}_b={breakup}"
output[key] = (
    Simulation1D(
        Settings1D(
            n_sd_per_gridbox=n_sd_per_gridbox,
            rho_times_w_1=rho_times_w,
            dt=dt,
            dz=dz,
            precip=precip,
            breakup=breakup,
        )
    )
    .run()
    .products
)

print("completed case 1")

# WITH BREAKUP
n_sd_per_gridbox = 1024
dt = 5 * si.s
dz = 50 * si.m

frag_scale_r = 30 * si.um
frag_scale_v = frag_scale_r**3 * 4 / 3 * np.pi

rho_times_w = 3 * si.m / si.s
precip = True
breakup = True
stochastic = False
key = f"rhow={rho_times_w}_p={precip}_b={breakup}_s={stochastic}"
settings = Settings1D(
    n_sd_per_gridbox=n_sd_per_gridbox,
    rho_times_w_1=rho_times_w,
    dt=dt,
    dz=dz,
    precip=precip,
    breakup=breakup,
)
settings.coalescence_efficiency = ConstEc(Ec=0.98)
settings.breakup_efficiency = ConstEb(Eb=1.0)
settings.fragmentation_function = Gaussian(
    mu=frag_scale_v, sigma=frag_scale_v / 2, vmin=(1 * si.um) ** 3, nfmax=20
)
settings.warn_breakup_overflow = False
other_label = ""
output[key] = Simulation1D(settings).run().products
print("completed case 2")

# WITH BREAKUP
n_sd_per_gridbox = 1024
dt = 5 * si.s
dz = 50 * si.m

rho_times_w = 3 * si.m / si.s
precip = True
breakup = True
stochastic = True
key = f"rhow={rho_times_w}_p={precip}_b={breakup}_s={stochastic}"
settings = Settings1D(
    n_sd_per_gridbox=n_sd_per_gridbox,
    rho_times_w_1=rho_times_w,
    dt=dt,
    dz=dz,
    precip=precip,
    breakup=breakup,
    stochastic_breakup=stochastic,
)
settings.warn_breakup_overflow = False
other_label = ""
output[key] = Simulation1D(settings).run().products
print("completed case 3")

# Save data
file = open("rainshaft_data_1024sd_5s_50m_Ec98.pkl", "wb")
pkl.dump(output, file)
file.close()
print("saved")
