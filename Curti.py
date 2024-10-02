'''
This module defines several useful functions for the analysis
of this work related to the paper Curti+(2020) 
https://ui.adsabs.harvard.edu/abs/2020MNRAS.491..944C/abstract
#
Functions:
    - get_Curti_FMR_params(sim,STARS_OR_GAS='GAS')
        Get FMR parameters for a Curti+2020 functional form of the FMR
        
    - thin_SFR_bin(mass,Z,SFR,ax,Z0)
        Deprecated (not used for publication)
#
Code written by: Alex Garcia, 2024
'''
### Standard Imports
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cmasher as cmr
### From this library
from helpers import (
    WHICH_SIM_TEX, get_one_redshift, get_medians, switch_sim,
    Curti_MZR, Curti_FMR
)

### Values from Curti+2020
# Curti_M0  = [9.81,9.77,9.74,9.75,9.96,10.12,10.14,10.22,10.25,10.33,10.40,10.46,10.48,10.33]
# Curti_SFR = [-0.75,-0.6,-0.45,-0.3,-0.15,0.0,0.15,0.3,0.45,0.6,0.75,0.95,1.05,1.2]

def get_Curti_FMR_params(sim,STARS_OR_GAS='GAS'):
    '''Get FMR parameters for a Curti+2020 functional form of the FMR
    
    Inputs:
    - sim (String): one of original, tng, eagle, or simba
    - STARS_OR_GAS (String): do this for stars or gas
    
    Returns:
    - (list): Best fit parameters for the FMR
    '''
    sim = sim.upper()
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)

    redshifts = np.arange(0,9)

    width = 0.1; step = 0.1; min_samp = 20

    snap = snapshots[0]

    star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,STARS_OR_GAS=STARS_OR_GAS)
    
    ### Do this for z=0
    snap = snapshots[0]
    star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                              STARS_OR_GAS=STARS_OR_GAS)
    
    MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                   width=width,
                                                   min_samp=min_samp)

    max_iter = 5000
    if sim == "SIMBA": ## SIMBA does not converge at low max_iter
        max_iter = int(1e6)
    params, cov = curve_fit(Curti_MZR, 10**MZR_M_real, MZR_Z_real,p0=[9.0,0.3,1.2,10**10.5],
                            maxfev=max_iter)
    
    Z0 = params[0]
    Z0_uncer = np.sqrt(np.diag(cov))[0]
    
    combined_data = np.vstack((10**MZR_M_real,real_SFR))
    
    Curti_FMR_wrapper = lambda X, gamma, beta, m0, m1: Curti_FMR(X, Z0, gamma, beta, m0, m1)
        
    params, cov = curve_fit(Curti_FMR_wrapper, combined_data, MZR_Z_real, p0=[0.28,1.2,10.11,0.56],
                            maxfev=max_iter)
        
    print('FMR')
    FMR_params = [Z0, *params]
    FMR_uncert = [Z0_uncer, *np.sqrt(np.diag(cov))]
    ps = ['Z0', 'gamma', 'beta', 'm0', 'm1']
    pretty_print = [f"{param:0.2f}" for param in FMR_params]
    for index, _ in enumerate(pretty_print):
        print(ps[index] + ': ' + _ + ' +/- ' + f"{FMR_uncert[index]:0.2f}")

    return FMR_params


def thin_SFR_bin(mass,Z,SFR,ax,Z0):
    '''Deprecated'''
    bin_width = 0.15 ## dex
    
    SFR = np.log10(SFR)
    
    SFR_bins = np.arange(np.percentile(SFR,5),np.percentile(SFR,95),bin_width)
    
    cmap = cmr.get_sub_cmap('cmr.pride', 0.2, 0.9, N=len(SFR_bins))
    newcolors = np.linspace(0, 1, len(SFR_bins))
    colors = [ cmap(x) for x in newcolors[::] ]
    
    M0 = []
    SFR_w_M0 = []
        
    for index in range(len(SFR_bins)-1):
        SFR_bin = SFR_bins[index]
        this_bin = (SFR > SFR_bin) & (SFR < SFR_bin + bin_width)
        
        this_mass = mass[this_bin]
        this_Z    =    Z[this_bin]
        this_SFR  =  SFR[this_bin]
        
        if sum(this_bin) < 3:
            continue
        
        MZR_M, MZR_Z, MZR_SFR = get_medians(this_mass,this_Z,this_SFR,width=width,min_samp=1)
                
        if len(MZR_M) > 0:
            bin_lbl = str(round(SFR_bin + width/2,2))
            ax.scatter(MZR_M, MZR_Z, color=colors[index],label=bin_lbl,s=30,alpha=0.5)
            
            combined_data = np.vstack((10**MZR_M, MZR_SFR))
            
            Curti_FMR_wrapper = lambda X, gamma, beta, m0, m1: Curti_FMR(X, Z0, gamma, beta, m0, m1)
            
            try:
                params, cov = curve_fit(Curti_FMR_wrapper,combined_data, MZR_Z, p0=[0.28,1.2,10.11,0.56],
                                        maxfev=10000)
                print(params)

                ax.plot(MZR_M, Curti_FMR(combined_data,Z0,*params), color=colors[index])
            except:
                print(sim,SFR_bin,'did not converge')
                continue
                
    return colors, M0, SFR_w_M0
