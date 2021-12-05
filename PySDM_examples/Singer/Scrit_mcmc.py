import numpy as np
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import os

from PySDM import Formulae
from PySDM.physics import si, constants as const
from PySDM.physics.surface_tension import compressed_film_ovadnevaite
from PySDM_examples.Singer.aerosol import AerosolBetaCary

os.environ['NUMBA_DISABLE_JIT'] = "1"

# parameter transformation so the MCMC parameters range from [-inf, inf]
# but the compressed film parameters are bounded appropriately
# sgm_org = [0,72.8] and delta_min = [0,inf]
def param_transform(mcmc_params):
    film_params = np.copy(mcmc_params)
    film_params[0] = const.sgm_w / (1 + np.exp(-1 * mcmc_params[0])) / si.mN * si.m 
    film_params[1] = np.exp(mcmc_params[1])
    return film_params

def negSS(r_wet, args):
    formulae, T, r_dry, kappa, f_org = args
    v_dry = formulae.trivia.volume(r_dry)
    v_wet = formulae.trivia.volume(r_wet)
    sigma = formulae.surface_tension.sigma(T, v_wet, v_dry, f_org)
    RH_eq = formulae.hygroscopicity.RH_eq(r_wet, T, kappa, r_dry**3, sigma)
    SS = (RH_eq - 1)*100
    return -1*SS

# evaluate the y-values of the model, given the current guess of parameter values
def get_model(params, args): 
    T, r_dry, OVF = args
    c = AerosolBetaCary(OVF)
    f_org = c.aerosol_modes_per_cc[0]['f_org']
    kappa = c.aerosol_modes_per_cc[0]['kappa']['Ovad']
    
    compressed_film_ovadnevaite.sgm_org = param_transform(params)[0] * si.mN / si.m
    compressed_film_ovadnevaite.delta_min = param_transform(params)[1] * si.nm
    formulae = Formulae(surface_tension='CompressedFilmOvadnevaite')
    
    Scrit, rcrit = np.zeros(len(r_dry)), np.zeros(len(r_dry))
    for i, rd in enumerate(r_dry):
        args = [formulae, T, r_dry[i], kappa[i], f_org[i]]
        res = minimize_scalar(negSS, args=args)
        Scrit[i], rcrit[i] = -1*res.fun, res.x
    
    return Scrit

# obtain the chi2 value of the model y-values given current parameters 
# vs. the measured y-values  
# calculate chi2 not log likelihood
def get_chi2(params, args, y, error): 
    model = get_model(params, args)
    chi2 = np.sum(((y-model)/error)**2)
    return chi2

# propose a new parameter set
# take a step in one paramter
# of random length in random direction
# with stepsize chosen from a normal distribution with width sigma
def propose_param(current_param, stepsize): 
    picker = int(np.floor(np.random.random(1)*len(current_param)))
    sigma = stepsize[picker]
    perturb_value = np.random.normal(0.0,sigma)

    try_param = np.zeros(len(current_param))
    try_param[~picker] = current_param[~picker]
    try_param[picker] = current_param[picker] + perturb_value
    
    try_param = np.copy(current_param)
    try_param[picker] = current_param[picker] + perturb_value
        
    return try_param, picker

#evaluate whether to step to the new trial value
def step_eval(params, stepsize, args, y, error):
    chi2_old = get_chi2(params, args, y, error)
    try_param, picker = propose_param(params, stepsize)
    chi2_try = get_chi2(try_param, args, y, error)
       
    # determine whether a step should be taken
    if chi2_try <= chi2_old:
        new_param = try_param
        accept_value = 1
    else:
        alpha = np.exp(chi2_old-chi2_try)
        r = np.random.random(1)
        if r < alpha:
            new_param = try_param
            accept_value = 1
        else:
            new_param = params
            accept_value = 0
    
    chi2_value = get_chi2(new_param, args, y, error)
    return new_param, picker, accept_value, chi2_value
    
#run the whole MCMC routine, calling the subroutines written above
def MCMC(params, stepsize, args, y, error, n_steps):
    param_chain = np.zeros((len(params),n_steps))
    accept_chain = np.empty((len(params),n_steps))
    accept_chain[:] = np.nan
    chi2_chain = np.zeros(n_steps)

    for i in np.arange(n_steps):
        param_chain[:,i], ind, accept_value, chi2_chain[i] = step_eval(params, stepsize, args, y, error)
        accept_chain[ind,i] = accept_value
        params = param_chain[:,i]
        
    return param_chain, accept_chain, chi2_chain