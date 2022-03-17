from matplotlib import pylab
import numpy as np
from corner import corner

from atmos_cloud_sim_uj_utils import show_plot
from kappa_mcmc import param_transform
from kappa_mcmc import get_model

def plot_param_chain(param_chain, args, title):
    T, r_dry, ovf, c, model = args
    p = param_transform(param_chain, model)

    if model == "CompressedFilmOvadnevaite":
        labels=["sgm_org", "delta_min"]
        fig, axes = pylab.subplots(2,1,figsize=(6,8))
        for i, ax in enumerate(axes.flatten()):
            ax.plot(p[i,:])
            ax.set_ylabel(labels[i])
            ax.grid()
    elif model == "CompressedFilmRuehl":
        labels=["A0", "C0", "sgm_min", "m_sigma"]
        fig, axes = pylab.subplots(2,2,figsize=(12,8))
        for i, ax in enumerate(axes.flatten()):
            ax.plot(p[i,:])
            ax.set_ylabel(labels[i])
            ax.grid()
    elif model == "SzyszkowskiLangmuir":
        labels=["A0", "C0", "sgm_min"]
        fig, axes = pylab.subplots(3,1,figsize=(6,12))
        for i, ax in enumerate(axes.flatten()):
            ax.plot(p[i,:])
            ax.set_ylabel(labels[i])
            ax.grid()
    else:
        raise AssertionError()
    pylab.tight_layout()
    
    modelname = model.split("CompressedFilm")[-1]
    aerosolname = c.__class__.__name__.split("Aerosol")[-1]
    pylab.savefig(aerosolname+"_"+title+"_"+modelname+"_chain.png", dpi=200, bbox_inches="tight")
    pylab.show()
        
def plot_corner(param_chain, args, title):
    T, r_dry, ovf, c, model = args
    data = param_transform(param_chain, model).T

    if model == "CompressedFilmOvadnevaite":
        labels=["sgm_org", "delta_min"]
    elif model == "CompressedFilmRuehl":
        labels=["A0", "C0", "sgm_min", "m_sigma"]
    elif model == "SzyszkowskiLangmuir":
        labels=["A0", "C0", "sgm_min"]
    else:
        raise AssertionError()
    
    pylab.rcParams.update({"font.size":12})
    figure = corner(data,
                    labels=labels,
                    label_kwargs={"fontsize": 12},
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt = ".1e",
                    title_kwargs={"fontsize": 12}
                    )
    
    modelname = model.split("CompressedFilm")[-1]
    aerosolname = c.__class__.__name__.split("Aerosol")[-1]
    pylab.savefig(aerosolname+"_"+title+"_"+modelname+"_corner.png", dpi=200, bbox_inches="tight")
    pylab.show()
    
def plot_ovf_kappa_fit(param_chain, args, errorx, datay, errory, title):
    T, r_dry, ovf, c, model = args
    
    # create aerosol
    dat = np.zeros((len(ovf),4))
    f_org = c.aerosol_modes_per_cc[0]['f_org']
    kappa = c.aerosol_modes_per_cc[0]['kappa'][model]

    pylab.figure(figsize=(10,6))

    # before
    kap_eff = get_model(param_chain[:,0], args)
    s = np.argsort(ovf)
    dat[:,0] = ovf[s]
    dat[:,2] = kap_eff[s]
    pylab.plot(ovf[s], kap_eff[s], 'b:', label="before")

    # after
    kap_eff2 = get_model(param_chain[:,-1], args)
    dat[:,3] = kap_eff2[s]
    pylab.plot(ovf[s], kap_eff2[s], 'r-', label="after")

    # data
    s = np.argsort(ovf)
    dat[:,1] = datay[s]
    pylab.errorbar(ovf, datay, yerr=errory, xerr=errorx, fmt='ko')

    pylab.legend()
    pylab.xlabel("Organic Volume Fraction")
    pylab.ylabel("$\kappa_{eff}$",fontsize=20)
    pylab.rcParams.update({"font.size":15})
    pylab.grid()
    pylab.tight_layout()
    
    modelname = model.split("CompressedFilm")[-1]
    aerosolname = c.__class__.__name__.split("Aerosol")[-1]
    pylab.savefig(aerosolname+"_"+title+"_"+modelname+"_fit.png", dpi=200, bbox_inches="tight")
    pylab.show()
    
def plot_keff(param_chain, args, datay, errory):
    T, r_dry, ovf, c, model = args
    
    # create aerosol
    dat = np.zeros((len(ovf),4))
    f_org = c.aerosol_modes_per_cc[0]['f_org']
    kappa = c.aerosol_modes_per_cc[0]['kappa'][model]

    pylab.figure(figsize=(10,6))

    # before
    kap_eff = get_model(param_chain[:,0], args)

    # after
    kap_eff2 = get_model(param_chain[:,-1], args)

    pylab.figure(figsize=(7,6))
    pylab.errorbar(kap_eff, datay, yerr=errory, fmt='bo', label="before")
    pylab.errorbar(kap_eff2, datay, yerr=errory, fmt='ro', label="after")
    pylab.legend()
    pylab.xlabel("$\kappa_{eff}$ modeled",fontsize=20)
    pylab.ylabel("$\kappa_{eff}$ measured",fontsize=20)
    pylab.plot([-0.05,0.55],[-0.05,0.55], 'k-')
    pylab.xlim([-0.05,0.55])
    pylab.ylim([-0.05,0.55])
    pylab.rcParams.update({"font.size":15})
    pylab.grid()
    pylab.show()