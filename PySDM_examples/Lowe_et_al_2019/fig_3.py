#!/usr/bin/env python
# coding: utf-8

# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM-examples.git/main?urlpath=PySDM_examples/Lowe_et_al_2019/fig_3.ipynb)
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PySDM-examples/blob/main/PySDM_examples/Lowe_et_al_2019/fig_3.ipynb)

# #### based on Fig. 3 from Lowe et al. 2019 (Nature Comm.)  "_Key drivers of cloud response to surface-active organics_"   
# https://doi.org/10.1038/s41467-019-12982-0

# In[1]:


import sys
if 'google.colab' in sys.modules:
    get_ipython().system('pip --quiet install atmos-cloud-sim-uj-utils')
    from atmos_cloud_sim_uj_utils import pip_install_on_colab
    pip_install_on_colab('PySDM-examples')


# In[2]:


from PySDM_examples.Lowe_et_al_2019 import Settings, Simulation
from PySDM_examples.Lowe_et_al_2019.aerosol import AerosolMarine, AerosolBoreal

#!pip install -e ~/Research/PySDM
import PySDM
print(PySDM.__file__)
from PySDM.initialisation.sampling import spectral_sampling as spec_sampling
from PySDM.physics import si

import numpy as np
import os
from matplotlib import pyplot
import matplotlib
from joblib import Parallel, delayed, parallel_backend
import numba


# In[3]:


numba.config.NUMBA_NUM_THREADS=1
rtol_x = 1e-3
rtol_thd = 1e-3


# In[4]:


CI = 'CI' in os.environ

updraft_list = np.geomspace(0.1, 10, 2 if CI else 5)
forg_list = np.linspace(0.1, 0.9, 2 if CI else 5)

subplot_list = ["a","b","c","d"]
# subplot_list = ["b"]
models = ('bulk', 'film')

Acc = {"a": 30, "b": 134, "c": 160, "d": 540}


# In[ ]:


def compute(key, settings):
    settings.rtol_x = rtol_x
    settings.rtol_thd = rtol_thd
    simulation = Simulation(settings)
    output = simulation.run()
    output['updraft'] = settings.w
    output['org_fraction'] = settings.aerosol.aerosol_modes_per_cc[0]['f_org']
    output['color'] = settings.aerosol.color
    return key, output

print(f'tasks scheduled: {len(models) * len(subplot_list) * len(forg_list) * len(updraft_list)}')
with parallel_backend('loky', n_jobs=-1):
    output = dict(Parallel(verbose=10)(
        delayed(compute)(subplot+f"_w{w:.2f}_f{Forg:.2f}_"+model, Settings(
            dz = 1 * si.m,
            # dt = 1.0 * si.s, 
            n_sd_per_mode = 25, 
            model = model,
            aerosol = {
                "a": AerosolMarine(Acc_Forg=Forg, Acc_N2=Acc["a"]), 
                "b": AerosolMarine(Acc_Forg=Forg, Acc_N2=Acc["b"]), 
                "c": AerosolBoreal(Acc_Forg=Forg, Acc_N2=Acc["c"]), 
                "d": AerosolBoreal(Acc_Forg=Forg, Acc_N2=Acc["d"])
            }[subplot],
            w = w * si.m / si.s,
            spectral_sampling = spec_sampling.ConstantMultiplicity
            #spectral_sampling = spec_sampling.Logarithmic
        ))
        for w in updraft_list
        for Forg in forg_list
        for subplot in subplot_list
        for model in models
    ))


# In[ ]:


# fig,axes = pyplot.subplots(len(subplot_list),len(updraft_list), sharex=False, sharey=True, figsize=(3*len(updraft_list),4*len(subplot_list)))
fig,axes = pyplot.subplots(1,len(updraft_list), sharex=False, sharey=True, figsize=(3*len(updraft_list),4))


for k,subplot in enumerate(subplot_list):
    for i,w in enumerate(updraft_list):
        for j,Forg in enumerate(forg_list):
            key = subplot+f"_w{w:.2f}_f{Forg:.2f}_"
            var = 'n_c_cm3'
            z = np.array(output[key+"film"]['z'])
            CDNC_film = np.array(output[key+"film"][var])
            CDNC_bulk = np.array(output[key+"bulk"][var])

            cmap = matplotlib.cm.get_cmap('Spectral')
            #ax = axes[k,i]
            ax = axes[i]

            ax.plot(CDNC_film, z, "--", color=cmap(Forg))
            ax.plot(CDNC_bulk, z, "-", color=cmap(Forg), label=f"{Forg:.2f}")

            if i == 0:
                ax.set_ylabel("Parcel displacement [m]")
                ax.set_title(subplot, loc="left", weight="bold")
            if i == len(updraft_list)-1 and k == 0:
                ax.legend(title="Forg", loc=2)
            if k == 0:
                ax.set_title(f"w = {w:.2f} m/s")
            if k == len(subplot_list)-1:
                ax.set_xlabel("CDNC [cm$^{-3}$]")

pyplot.savefig("fig3-parcel-profiles.png",dpi=200)
pyplot.show()


# In[ ]:


# fig,axes = pyplot.subplots(2,2,figsize=(10,10),sharex=True,sharey=False)

# f0 = open("PySDM_Marine_0_0org.txt","w")
# f1 = open("PySDM_Marine_1_0org.txt","w")

# f0.write("parcel disp = 200.00 \n")
# f0.write("updraft (m/s), forg, CDNC (cm-3) \n")
# f1.write("parcel disp = 200.00 \n")
# f1.write("updraft (m/s), forg, CDNC (cm-3) \n")

# for k,subplot in enumerate(subplot_list):
#     for i,w in enumerate(updraft_list):
#         for j,Forg in enumerate(forg_list):
#                 key = subplot+"_w{:.2f}_f{:.2f}_".format(w,Forg)
#                 var = 'n_c_cm3'
#                 z = np.array(output[key+"film"]['z'])
#                 CDNC_film = np.array(output[key+"film"][var])
#                 CDNC_bulk = np.array(output[key+"bulk"][var])
                
#                 ax = axes.flatten()[k]
#                 ax.set_title(subplot, loc="left", weight="bold")
                
#                 cmap = matplotlib.cm.get_cmap('Spectral')
#                 if i == 0:
#                     ax.plot(w, CDNC_bulk[-1], 'o', color=cmap(Forg), label="{:.2f}".format(Forg))
#                     ax.plot(w, CDNC_film[-1], '*', color=cmap(Forg))
#                 else:
#                     ax.plot(w, CDNC_bulk[-1], 'o', color=cmap(Forg))
#                     ax.plot(w, CDNC_film[-1], '*', color=cmap(Forg))
                
#                 if subplot == "b":
#                     f0.write("{:.2e}, {:.2f}, {:.2f} \n".format(w, Forg, CDNC_bulk[-1]))
#                     f1.write("{:.2e}, {:.2f}, {:.2f} \n".format(w, Forg, CDNC_film[-1]))
                
#                 ax.set_xscale("log")
#                 if k == 0:
#                     ax.legend(title="Forg")
#                     ax.set_ylabel("CNDC [cm$^{-3}$]")
#                 if k == 2:
#                     ax.set_ylabel("CNDC [cm$^{-3}$]")
#                 if k > 1:
#                     ax.set_xlabel("updraft velocity [m/s]")
                
# #pyplot.savefig("fig3-parcel-profiles.png",dpi=200)
# pyplot.show()

# f0.close()
# f1.close()

# #########################

# fig,axes = pyplot.subplots(2,2,figsize=(10,10),sharex=True,sharey=False)

# for k,subplot in enumerate(subplot_list):
#     for i,w in enumerate(updraft_list):
#         for j,Forg in enumerate(forg_list):
#                 key = subplot+"_w{:.2f}_f{:.2f}_".format(w,Forg)
#                 var = 'n_c_cm3'
#                 z = np.array(output[key+"film"]['z'])
#                 CDNC_film = np.array(output[key+"film"][var])
#                 CDNC_bulk = np.array(output[key+"bulk"][var])
                
#                 ax = axes.flatten()[k]
#                 ax.set_title(subplot, loc="left", weight="bold")
                
#                 cmap = matplotlib.cm.get_cmap('Spectral')
#                 if i == 0:
#                     ax.plot(w, CDNC_film[-1] - CDNC_bulk[-1], 'o', color=cmap(Forg), label="{:.2f}".format(Forg))
#                 else:
#                     ax.plot(w, CDNC_film[-1] - CDNC_bulk[-1], 'o', color=cmap(Forg))
                
#                 ax.set_xscale("log")
#                 if k == 0:
#                     ax.legend(title="Forg")
#                     ax.set_ylabel(r"$\Delta$CNDC [cm$^{-3}$]")
#                 if k == 2:
#                     ax.set_ylabel(r"$\Delta$CNDC [cm$^{-3}$]")
#                 if k > 1:
#                     ax.set_xlabel("updraft velocity [m/s]")
                
# #pyplot.savefig("fig3-parcel-profiles.png",dpi=200)
# pyplot.show()

# #########################

# fig,axes = pyplot.subplots(2,2,figsize=(10,10),sharex=True,sharey=False)

# for k,subplot in enumerate(subplot_list):
#     for i,w in enumerate(updraft_list):
#         for j,Forg in enumerate(forg_list):
#                 key = subplot+"_w{:.2f}_f{:.2f}_".format(w,Forg)
#                 var = 'n_c_cm3'
#                 z = np.array(output[key+"film"]['z'])
#                 CDNC_film = np.array(output[key+"film"][var])
#                 CDNC_bulk = np.array(output[key+"bulk"][var])
                
#                 ax = axes.flatten()[k]
#                 ax.set_title(subplot, loc="left", weight="bold")
                
#                 cmap = matplotlib.cm.get_cmap('Spectral')
#                 if i == 0:
#                     ax.plot(w, (CDNC_film[-1] - CDNC_bulk[-1]) / CDNC_bulk[-1] * 100.0, 'o', color=cmap(Forg), label="{:.2f}".format(Forg))
#                 else:
#                     ax.plot(w, (CDNC_film[-1] - CDNC_bulk[-1]) / CDNC_bulk[-1] * 100.0, 'o', color=cmap(Forg))
                
#                 ax.set_xscale("log")
#                 if k == 0:
#                     ax.legend(title="Forg")
#                     ax.set_ylabel(r"$\Delta$CNDC [%]")
#                 if k == 2:
#                     ax.set_ylabel(r"$\Delta$CNDC [%]")
#                 if k > 1:
#                     ax.set_xlabel("updraft velocity [m/s]")
                
# #pyplot.savefig("fig3-parcel-profiles.png",dpi=200)
# pyplot.show()


# In[ ]:


fig, axes = pyplot.subplots(2,2, sharex=True, sharey=True, figsize=(14,10))

for subplot in subplot_list:
    dCDNC = np.zeros((len(updraft_list), len(forg_list)))
    for i,w in enumerate(updraft_list):
        for j,Forg in enumerate(forg_list):
            key = subplot+f"_w{w:.2f}_f{Forg:.2f}_"
            var = 'n_c_cm3'
            z = np.array(output[key+"film"]['z'])
            wz = np.where(z == z[-1])[0][0]
            CDNC_film = np.array(output[key+"film"][var])[wz]
            CDNC_bulk = np.array(output[key+"bulk"][var])[wz]
            dCDNC[i,j] = (CDNC_film - CDNC_bulk) / CDNC_bulk * 100.0
            #print(w, Forg, CDNC_bulk, CDNC_film, dCDNC[i,j])

    if subplot == "a":
        ax = axes[0,0]
        ax.set_title("MA Accum. mode conc. N$_2 = 30$ cm$^{-3}$", fontsize=13, loc="right")
        ax.contour(forg_list, updraft_list, dCDNC, levels=[10,25], colors=["#1fa8f2","#4287f5"], 
                       linestyles=[":","--"], linewidths=4)
        p = ax.contourf(forg_list, updraft_list, dCDNC, cmap="Blues", levels=np.linspace(0,90,11))
    if subplot == "b":
        ax = axes[0,1]
        ax.set_title("MA Accum. mode conc. N$_2 = 135$ cm$^{-3}$", fontsize=13, loc="right")
        ax.contour(forg_list, updraft_list, dCDNC, levels=[10,25], colors=["#1fa8f2","#4287f5"], 
                       linestyles=[":","--"], linewidths=4)
        p = ax.contourf(forg_list, updraft_list, dCDNC, cmap="Blues", levels=np.linspace(0,90,11))
    if subplot == "c":
        ax = axes[1,0]
        ax.set_title("HYY Accum. mode conc. N$_2 = 160$ cm$^{-3}$", fontsize=13, loc="right")
        ax.contour(forg_list, updraft_list, dCDNC, levels=[10,25], colors=["#04c753","#157d3f"], 
                       linestyles=[":","--"], linewidths=4)
        p = ax.contourf(forg_list, updraft_list, dCDNC, cmap="Greens", levels=np.linspace(0,65,11))
    if subplot == "d":
        ax = axes[1,1]
        ax.set_title("HYY Accum. mode conc. N$_2 = 540$ cm$^{-3}$", fontsize=13, loc="right")
        ax.contour(forg_list, updraft_list, dCDNC, levels=[10,25], colors=["#04c753","#157d3f"], 
                       linestyles=[":","--"], linewidths=4)
        p = ax.contourf(forg_list, updraft_list, dCDNC, cmap="Greens", levels=np.linspace(0,65,11))
        
    ax.set_title(subplot,weight="bold",loc="left")
    if subplot in ("c", "d"):
        ax.set_xlabel("Organic mass fraction")
    ax.set_yscale("log")
    ax.set_yticks([0.1,1,10])
    ax.set_yticklabels(["0.1","1","10"])
    if subplot in ("a", "c"):
        ax.set_ylabel("Updraft [ms$^{-1}$]")
    pyplot.colorbar(p, ax=ax, label=r"$\Delta_{CDNC}$ [%]")

pyplot.rcParams.update({'font.size': 15})
pyplot.savefig("fig3.png", dpi=200)
pyplot.show()


# In[ ]:




