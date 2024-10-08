'''
This module defines several useful functions for the analysis
of this work
#
Functions:
    - get_epsilon_evo(sim,n_bootstrap=5)
        Calculate the evolution of epsilon (a fitting parameter)
        across redshifts for a given simulation
        
    - get_avg_relations(sim,redshift)
        Get the z=0 MZR (w/ SFRs) for each simulation
        
    - get_rnd_sample(sim,redshift)
        Gets a random sample of galaxies for bootstrapping
        
    (all others deprecated; not recommended for usage)
#
Code written by: Alex Garcia, 2024
'''
### Standard Imports
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import t
### From this library
from helpers import (
    get_all_redshifts, WHICH_SIM_TEX, get_z0_alpha,
    get_medians, linear, fourth_order
)
 
def get_epsilon_evo(sim,n_bootstrap=5):
    '''Calculate the evolution of epsilon (a fitting parameter)
    across redshifts for a given simulation
    
    Inputs:
    - sim (String): name of the simulation, e.g., "SIMBA"
    - n_bootstrap (int): number of bootstrap samples to determine uncertainty
    
    Returns:
    - (ndarray): Mean epsilon evolution at each redshift
    - (ndarray): Lower bound (16th percentile) of epsilon
    - (ndarray): Upper bound (84th percentile) of epsilon
    - (list): All epsilon values from bootstrap sampling
    '''
    sim = sim.upper()
    z0_mbins, z0_zbins, z0_Sbins = None, None, None
    z0_SFMS, z0_MZR = None, None
    
    unique = np.arange(0,9)
    
    epsilon_evo  = np.zeros(len(unique))
    evo_min    = np.zeros(len(unique))
    evo_max    = np.zeros(len(unique))
    all_epsilons = []
    
    for index, redshift in enumerate(unique):
        if index == 0:
            mbins, zbins, Sbins = get_avg_relations(sim,redshift)
            
            z0_mbins, z0_zbins, z0_Sbins = mbins, zbins, Sbins
            
            z0_SFMS = interp1d(z0_mbins, z0_Sbins, fill_value='extrapolate')
            z0_MZR  = interp1d(z0_mbins, z0_zbins, fill_value='extrapolate')
            continue
        
        mins = np.zeros(n_bootstrap)
        
        for bootstrap_index in tqdm(range(n_bootstrap)):
            this_mbins, this_zbins, this_Sbins = get_rnd_sample(sim,redshift)
            
            epsilons = np.linspace(0,1,100)
            MSE = np.zeros(len(epsilons))

            for epsilon_index, epsilon in enumerate(epsilons):
                Z_pred = -epsilon * np.log10( this_Sbins / z0_SFMS(this_mbins) ) + z0_MZR(this_mbins)
                MSE[epsilon_index] = sum((this_zbins - Z_pred)**2) / len(Z_pred)
            
            mins[bootstrap_index] = epsilons[np.argmin(MSE)]
        all_epsilons.extend(mins)
        lower = np.percentile(mins,16)
        upper = np.percentile(mins,84)
        
        epsilon_evo[index] = np.mean(mins)
        evo_min[index] = lower
        evo_max[index] = upper
        print( ("$" + f"{np.mean(mins):0.2f}" + "_{" + f"{lower:0.2f}" + "}^{" + f"{upper:0.2f}" + "}$") )
        
    return epsilon_evo, evo_min, evo_max, all_epsilons

def get_avg_relations(sim,redshift):
    '''Get the z=0 MZR (w/ SFRs) for each simulation
    
    Inputs:
    - sim (String): simulation name
    - redshift (int): redshift of interest
    
    Outputs:
    - calls get_medians from helpers
    '''
    star_mass, SFR, Z, redshifts = get_all_redshifts(sim, False)
    
    mask = redshifts == redshift
    current_m = star_mass[mask]
    current_Z = Z[mask]
    current_S = SFR[mask]

    return get_medians(current_m, current_Z, current_S)


def get_rnd_sample(sim,redshift):
    '''Gets a random sample of galaxies for bootstrapping
    
    Inputs:
    - sim (String): which simulation
    - redshift (int): which redshift
    
    Outputs:
    - calls get_medians() with random sample
    '''
    star_mass, SFR, Z, redshifts = get_all_redshifts(sim, False)
    
    mask = redshifts == redshift
    current_m = star_mass[mask]
    current_Z = Z[mask]
    current_S = SFR[mask]

    n_samples = len(current_m)
            
    this_mask = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)

    current_S = current_S[this_mask]
    current_m = current_m[this_mask]
    current_Z = current_Z[this_mask]
    
    return get_medians(current_m, current_Z, current_S)

if __name__ == "__main__":
    print('Hello World')

########################################
######### DEPRECATED GRAVEYARD #########
########################################
    
def get_alpha_evo(sim,function=linear):
    '''Deprecated'''
    sim = sim.upper()
    z0_mbins, z0_zbins, z0_Sbins = None, None, None
    
    redshifts = np.arange(0,9)
    
    alpha_evo = np.zeros(len(redshifts))
    evo_min   = np.zeros(len(redshifts))
    evo_max   = np.zeros(len(redshifts))
    
    if sim == "SIMBA": ## Exclude z=8 in SIMBA
        redshifts = redshifts[:-1]
        alpha_evo[-1] = np.nan
        evo_min[-1] = np.nan
        evo_max[-1] = np.nan
    
    for index, redshift in enumerate(redshifts):
        mbins, zbins, Sbins = get_avg_relations(sim, redshift)
        
        if index == 0:
            z0_mbins, z0_zbins, z0_Sbins = mbins, zbins, Sbins
            continue
            
        alphas = np.linspace(-1,1,200)
        disp   = np.zeros(len(alphas))

        all_mbins = np.append(mbins,z0_mbins)
        all_zbins = np.append(zbins,z0_zbins)
        all_Sbins = np.append(Sbins,z0_Sbins)
        
        min_alpha, *params = get_z0_alpha(sim,function=function)
        
        for j, alpha in enumerate(alphas):
            mu = all_mbins - alpha * np.log10(all_Sbins)
            
            # params, cov = curve_fit(function,mu,all_zbins)
            interp_line = function(mu, *params)
            # params = np.polyfit(mu, all_zbins, 1)
            # interp_line = np.polyval(params, mu)
            
            disp[j] = np.std( np.abs(all_zbins) - np.abs(interp_line) )
            
        argmin = np.argmin(disp)
        min_alpha = alphas[argmin]
        min_disp  = disp[argmin]
        
        width = 1.05 * min_disp
        
        within_uncertainty = alphas[ (disp < width) ]

        min_uncertain = within_uncertainty[0]
        max_uncertain = within_uncertainty[-1] 
        
        alpha_evo[index] = min_alpha
        evo_min[index] = min_uncertain
        evo_max[index] = max_uncertain
        
    return alpha_evo, evo_min, evo_max