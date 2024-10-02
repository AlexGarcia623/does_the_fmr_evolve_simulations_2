'''
This module defines several useful functions for the analysis
of this work
#
Functions:
    - format_func(value, tick_number)
        Helper for formatting axes
        
    - make_MZR_prediction_fig(sim,all_z_fit,ax_real,ax_fake,ax_offsets,
                              STARS_OR_GAS="gas",
                              function = linear,
                              THRESHOLD = -5.00E-01,
                              width = 0.1, step = 0.1,
                              min_samp = 20)
        Populate MZR, MZR predictions, and offsets for given simulation
        
    - make_MZR_prediction_fig_Curti(sim,all_z_fit,ax_real,ax_fake,ax_offsets,
                                    STARS_OR_GAS="gas",savedir="./",
                                    function = linear,
                                    THRESHOLD = -5.00E-01,
                                    width = 0.1, step = 0.1,
                                    min_samp = 20)
        Populate MZR, MZR predictions, and offsets for given simulation.
        This one uses the Curti+2020 MZR/FMR functional forms
        
    - make_MZR_prediction_fig_epsilon(sim,ax_real,ax_fake,ax_offsets,
                                      STARS_OR_GAS="gas",savedir="./",
                                      function = linear,
                                      THRESHOLD = -5.00E-01,
                                      width = 0.1, step = 0.1,
                                      min_samp = 20)
        Populate MZR, MZR predictions, and offsets for given simulation.
        This uses a variable epsilon value
        
    - make_MZR_fig(sim,ax,STARS_OR_GAS="gas",
                   THRESHOLD = -5.00E-01,
                   width = 0.1, step = 0.1,
                   min_samp = 20, bin_index=0)
        Get each simulation's MZR
        
    - make_SFMS_fig(sim,ax,STARS_OR_GAS="gas",
                    THRESHOLD = -5.00E-01,
                    width = 0.1, step = 0.1,
                    min_samp = 20)
        Get each simulation's SFMS
        
    (all others deprecated; not recommended for usage)
#
Code written by: Alex Garcia, 2024
'''
### Standard Imports
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import cmasher as cmr
### From this library
from helpers import (
    WHICH_SIM_TEX, get_z0_alpha, get_all_redshifts,
    get_one_redshift, get_medians, switch_sim,
    linear, fourth_order, Curti_FMR
)
from alpha_types import (
    get_avg_relations, get_epsilon_evo
)
from Curti import (
    get_Curti_FMR_params
)


def format_func(value, tick_number):
    '''Helper for formatting axes'''
    return "{:.1f}".format(value)

def make_MZR_prediction_fig(sim,all_z_fit,ax_real,ax_fake,ax_offsets,
                            STARS_OR_GAS="gas",
                            function = linear,
                            THRESHOLD = -5.00E-01,
                            width = 0.1, step = 0.1,
                            min_samp = 20):
    '''Populate MZR, MZR predictions, and offsets for given simulation
    
    Inputs:
    - sim (String): name of simulation
    - all_z_fit (boolean): DEPRECATED... = False
    - ax_real (matplotlib axes): axis to plot "real" MZR data
    - ax_fake (matplotlib axes): axis to plot predicted MZR data
    - ax_offsets (matplotlib axes): axis to plot "real" - predicted MZR offsets
    - STARS_OR_GAS (String): define gas-phase or stellar metallicity
    - function (python function): fitting routine
    - THRESHOLD (float): impacts selection criteria
    - width (float): Width of mass bins
    - step (float): amount to increase mass bins by
    - min_sample (int): minimum number of galaxies in a bin
    
    Returns:
    - (list): The cmasher colors used to make the plot
    - (list): The Mean Squared Error of offsets
    '''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    all_z_fit = False
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    star_mass_all, SFR_all, Z_use_all, redshifts_all = get_all_redshifts(sim,all_z_fit,THRESHOLD=THRESHOLD)

    if function == Curti_FMR: ## Curti has its own function
        return
    
    min_alpha, *params = get_z0_alpha(sim, function=function)
    best_mu = star_mass_all - min_alpha*np.log10(SFR_all)
    plot_mu = np.linspace(np.min(best_mu),np.max(best_mu),100)
    best_line = function(plot_mu, *params)
    
    unique = np.unique(redshifts_all)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    
    sum_residuals = 0
    n_resids = 0
    MSE = 0
    
    z0_MZR = None
    
    for index, snap in enumerate(snapshots):
        #star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
        #                                          STARS_OR_GAS=STARS_OR_GAS)
        mask = (redshifts_all == index)
        star_mass = star_mass_all[mask]
        Z_true    = Z_use_all[mask]
        SFR       = SFR_all[mask]

        MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                       width=width,
                                                       min_samp=min_samp)
        if index == 0: ## save z=0 MZR
            z0_MZR = interp1d(MZR_M_real, MZR_Z_real, fill_value='extrapolate')
        
        ### Plot 3 different types
        color = colors[index]
        
        mu = MZR_M_real - min_alpha * np.log10(real_SFR)
        
        lw = 3
        
        ax_real.plot( MZR_M_real, MZR_Z_real, color=color,
                      label=r'$z=%s$' %index, lw=lw )
        
        mu = MZR_M_real - min_alpha * np.log10(real_SFR)
        MZR_Z_fake = function(mu, *params)
        
        ax_fake.plot( MZR_M_real, MZR_Z_fake, color=color,
                      label=r'$z=%s$' %index, lw=lw )
        
        offset = MZR_Z_real - MZR_Z_fake
        
        sum_residuals += sum(offset**2)
        n_resids += len(offset)
        
        ax_offsets.plot( MZR_M_real, offset, color=color,
                         label=r'$z=%s$' %index, lw=lw )
        
        ### Calculate the Mean Squared Error
        if index == len(snapshots) - 1:
            MSE = sum_residuals / n_resids
            print(f'\tMSE: {MSE:.3f} (dex)^2')
            txt_loc_x = 0.075
            ha = 'left'
            ax_offsets.text( txt_loc_x, 0.07,
                             r'$\xi = \;$' + fr"${MSE:0.3f}$" + r'$\;({\rm dex })^2$',
                             transform=ax_offsets.transAxes, fontsize=16, ha=ha )
        
    return colors, MSE

def make_MZR_prediction_fig_Curti(sim,all_z_fit,ax_real,ax_fake,ax_offsets,
                                  STARS_OR_GAS="gas",savedir="./",
                                  function = linear,
                                  THRESHOLD = -5.00E-01,
                                  width = 0.1, step = 0.1,
                                  min_samp = 20):
    '''Populate MZR, MZR predictions, and offsets for given simulation.
    This one uses the Curti+2020 MZR/FMR functional forms
    
    Inputs:
    - sim (String): name of simulation
    - all_z_fit (boolean): DEPRECATED... = False
    - ax_real (matplotlib axes): axis to plot "real" MZR data
    - ax_fake (matplotlib axes): axis to plot predicted MZR data
    - ax_offsets (matplotlib axes): axis to plot "real" - predicted MZR offsets
    - STARS_OR_GAS (String): define gas-phase or stellar metallicity
    - function (python function): fitting routine (should be Curti_FMR routine)
    - THRESHOLD (float): impacts selection criteria
    - width (float): Width of mass bins
    - step (float): amount to increase mass bins by
    - min_sample (int): minimum number of galaxies in a bin
    
    Returns:
    - (list): The cmasher colors used to make the plot
    - (list): The Mean Squared Error of offsets
    '''
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,all_z_fit,THRESHOLD=THRESHOLD)

    if function != Curti_FMR: 
        return
    
    params = get_Curti_FMR_params(sim)
    
    unique = np.unique(redshifts)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    
    sum_residuals = 0
    n_resids = 0
    MSE = 0
    
    z0_MZR = None
    
    for index, snap in enumerate(snapshots):
        
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS)
        
        MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                       width=width,
                                                       min_samp=min_samp)
        
        if index == 0:
            z0_MZR = interp1d(MZR_M_real, MZR_Z_real, fill_value='extrapolate')
        
        color = colors[index]
                
        lw = 3
        
        ax_real.plot( MZR_M_real, MZR_Z_real, color=color,
                      label=r'$z=%s$' %index, lw=lw )
                
        combined_X = np.vstack((10**MZR_M_real,real_SFR))
        Z0 = params[0]
        Curti_FMR_wrapper = lambda X, gamma, beta, m0, m1: Curti_FMR(X, Z0, gamma, beta, m0, m1)
        MZR_Z_fake = Curti_FMR_wrapper(combined_X, *params[1:])
        
        ax_fake.plot( MZR_M_real, MZR_Z_fake, color=color,
                      label=r'$z=%s$' %index, lw=lw )
        
        offset = MZR_Z_real - MZR_Z_fake
        
        sum_residuals += sum(offset**2)
        n_resids += len(offset)
        
        ax_offsets.plot( MZR_M_real, offset, color=color,
                         label=r'$z=%s$' %index, lw=lw )
        
        if index == len(snapshots) - 1:
            MSE = sum_residuals / n_resids
            print(f'\tMSE: {MSE:.3f} (dex)^2')
            txt_loc_x = 0.075
            ha = 'left'
            if sim == "TNG":
                txt_loc_x = 0.95
                ha ='right'
            ax_offsets.text( txt_loc_x, 0.07,
                             r'$\xi = \;$' + fr"${MSE:0.3f}$" + r'$\;({\rm dex })^2$',
                             transform=ax_offsets.transAxes, fontsize=16, ha=ha )

    return colors, MSE

def make_MZR_prediction_fig_epsilon(sim,ax_real,ax_fake,ax_offsets,
                                    STARS_OR_GAS="gas",savedir="./",
                                    function = linear,
                                    THRESHOLD = -5.00E-01,
                                    width = 0.1, step = 0.1,
                                    min_samp = 20):
    '''Populate MZR, MZR predictions, and offsets for given simulation.
    This uses a variable epsilon value
    
    Inputs:
    - sim (String): name of simulation
    - all_z_fit (boolean): DEPRECATED... = False
    - ax_real (matplotlib axes): axis to plot "real" MZR data
    - ax_fake (matplotlib axes): axis to plot predicted MZR data
    - ax_offsets (matplotlib axes): axis to plot "real" - predicted MZR offsets
    - STARS_OR_GAS (String): define gas-phase or stellar metallicity
    - function (python function): fitting routine
    - THRESHOLD (float): impacts selection criteria
    - width (float): Width of mass bins
    - step (float): amount to increase mass bins by
    - min_sample (int): minimum number of galaxies in a bin
    
    Returns:
    - (list): The cmasher colors used to make the plot
    - (list): The Mean Squared Error of offsets
    '''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    star_mass, SFR, Z_true, redshifts = get_all_redshifts(sim,False,THRESHOLD=THRESHOLD)
    
    unique = np.unique(redshifts)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    
    sum_residuals = 0
    n_resids = 0
    MSE = 0
    
    z0_MZR = None
    
    delta_evo, *_ = get_epsilon_evo(sim)

    z0_SFMS, z0_MZR = None, None

    sum_residuals = 0
    n_data = 0

    min_alpha, *_ = get_z0_alpha(sim,function=linear)
    
    for jjj, redshift in enumerate(unique):
        if sim == "SIMBA" and redshift > 7:
            continue
        this_redshift = redshift == redshifts
        mbins, zbins, Sbins = get_medians(star_mass[this_redshift],
                                          Z_true[this_redshift],
                                          SFR[this_redshift],
                                          width=width,
                                          min_samp=min_samp)

        if jjj == 0:
            z0_mbins, z0_zbins, z0_Sbins = mbins, zbins, Sbins
            z0_mu = z0_mbins - min_alpha * np.log10(z0_Sbins)
            params, cov = curve_fit(function, z0_mu, z0_zbins)
            
            z0_zpred = function(z0_mu, *params)
            
            z0_SFMS = interp1d(mbins, Sbins, fill_value='extrapolate')
            z0_MZR  = interp1d(z0_mbins, z0_zpred, fill_value='extrapolate')

        lw = 3
        
        ax_real.plot(mbins, zbins, color=colors[jjj],
                    label="$z=%s$" %jjj, lw=lw)

        Z_pred = -delta_evo[jjj] * np.log10( Sbins / z0_SFMS(mbins) ) + z0_MZR(mbins)

        ax_fake.plot(mbins, Z_pred, color=colors[jjj],
                    label="$z=%s$" %jjj, lw=lw)

        residuals = zbins - Z_pred
        sum_residuals += sum(residuals**2)
        n_data += len(residuals)

        ax_offsets.plot(mbins, residuals, color=colors[jjj],
                        label="$z=%s$" %jjj, lw=lw)

    MSE = sum_residuals / n_data
    ax_offsets.text(0.075, 0.07,
                r'$\xi = \;$' + fr"${MSE:0.3f}$" + r'$\;({\rm dex })^2$',
                transform=ax_offsets.transAxes, fontsize=16)
        
    return colors, MSE
    
def make_MZR_fig(sim,ax,STARS_OR_GAS="gas",
                 THRESHOLD = -5.00E-01,
                 width = 0.1, step = 0.1,
                 min_samp = 20, bin_index=0):
    '''Get each simulation's MZR
    
    Inputs:
    - sim (String): name of simulation
    - ax (matplotlib axes): axis to plot data
    - STARS_OR_GAS (String): define gas-phase or stellar metallicity
    - THRESHOLD (float): impacts selection criteria
    - width (float): Width of mass bins
    - step (float): amount to increase mass bins by
    - min_sample (int): minimum number of galaxies in a bin
    - bin_index (int): index of bin to do inset plot of
    
    Returns:
    - (list): The cmasher colors used to make the plot
    '''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,False,THRESHOLD=THRESHOLD)
    
    unique = np.unique(redshifts)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    
    ## Holders for minimum SFR at each redshift
    Z_min_bin = []
    m_min = 0
    
    ## Data for Arnab
    y_dat_8   = []
    y_dat_9   = []
    y_dat_10  = []
    y_dat_105 = [] 
    
    for index, snap in enumerate(snapshots):
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS,snap2z=snap2z)
        
        MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                       width=width,
                                                       min_samp=min_samp)
        
        MZR_M_real -= width/2 ## Center bins
        
        ## Append lowest mass bin, if it exists
        if bin_index > len(MZR_Z_real) - 1: ## This is for video purposes
            Z_min_bin.append(np.nan)
        elif (len(MZR_Z_real) > 0):
            Z_min_bin.append(MZR_Z_real[bin_index])
        else:
            Z_min_bin.append(np.nan)
        
        if index == 0:
            m_min = MZR_M_real[bin_index]
        
        color = colors[index]
        lw = 2.5
        ax.plot( MZR_M_real, MZR_Z_real, color=color,
                      label=r'$z=%s$' %index, lw=lw )
    
        ## Data for Arnab
        if len(MZR_M_real > 0):
            if sim != 'SIMBA':
                y_dat_8.append( MZR_Z_real[ np.argmin(np.abs(MZR_M_real - 8)) ] )
            
            if np.max(MZR_M_real) > 8.9:
                y_dat_9.append( MZR_Z_real[ np.argmin(np.abs(MZR_M_real - 9)) ] )
            else:
                y_dat_9.append( np.nan )
                
            if np.max(MZR_M_real) > 9.9:
                y_dat_10.append( MZR_Z_real[ np.argmin(np.abs(MZR_M_real - 10)) ] )
            else:
                y_dat_10.append( np.nan )
                
            if np.max(MZR_M_real) > 10.4:
                y_dat_105.append(MZR_Z_real[np.argmin(np.abs(MZR_M_real - 10.5))])
            else:
                y_dat_105.append( np.nan )
    ## Data for Arnab
    if sim != 'SIMBA':
        np.save(f'./arnab_data/{sim}_8_Fig8.npy',y_dat_8)
    np.save(f'./arnab_data/{sim}_9_Fig8.npy',y_dat_9)
    np.save(f'./arnab_data/{sim}_10_Fig8.npy',y_dat_10)
    np.save(f'./arnab_data/{sim}_105_Fig8.npy',y_dat_105)
    if sim == "SIMBA":
        np.save('./arnab_data/x_axis_both.npy',np.arange(0,9))
    
    ## Get difference between z=0 minimum metallicity and all others
    Z_min_bin_no_subtract = np.array(Z_min_bin)
    Z_min_bin -= Z_min_bin_no_subtract[0]

    xstart = 10.25
    ystart = 7.05
    xlen = 1.6
    ylen = 1
        
    ax_inset = ax.inset_axes([xstart, ystart, xlen, ylen], transform=ax.transData)
    
    ax_inset.scatter(unique, Z_min_bin, c=colors, s=10)
        
    ax_inset.spines['bottom'].set_linewidth(1); ax_inset.spines['top'].set_linewidth(1)
    ax_inset.spines['left'].set_linewidth(1)  ; ax_inset.spines['right'].set_linewidth(1)
        
    ax_inset.tick_params(axis='both', which='major', length=3, width=1.5)
    ax_inset.tick_params(axis='both', which='minor', length=2, width=1)
    
    ax_inset.tick_params(axis='y', labelsize=10)
    ax_inset.tick_params(axis='x', which='both', top=True, labelsize=10)
    
    ax_inset.set_ylim(-1.1,0.2)
    ax_inset.set_yticks([-1,-0.5,0])
    ax_inset.minorticks_on()
    ax_inset.set_xlim(-1,9)
    
    _y_ = 0.5
    if sim == "SIMBA":
        _y_ = 0.4
    ax_inset.text(-0.4,_y_,r'${\rm Offset~(dex)}$', fontsize=10,
                  transform=ax_inset.transAxes, va='center', rotation=90)
    ax_inset.text(0.5,-0.3,r'${\rm Redshift}$', fontsize=10,
                  transform=ax_inset.transAxes, ha='center')
    
    ax_inset.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax_inset.set_xticks([0,4,8])
    
    ymin = np.nanmin(Z_min_bin_no_subtract)
    ymax = np.nanmax(Z_min_bin_no_subtract)
    
    x = np.linspace(m_min, xstart, 100)
    
    slope1 = (ystart - ymin) / (xstart - m_min)
    slope2 = (ystart + ylen - ymax) / (xstart - m_min)
    
    y1 = slope1 * (x - xstart) + ystart
    y2 = slope2 * (x - xstart) + ystart + ylen
    
    ax.plot(x, y1, color='k', alpha=0.5, lw=1)
    ax.plot(x, y2, color='k', alpha=0.5, lw=1)
    ax.vlines(x=m_min, ymin=ymin, ymax=ymax, color='k', lw=1)
    
    return colors

def make_SFMS_fig(sim,ax,STARS_OR_GAS="gas",
                  THRESHOLD = -5.00E-01,
                  width = 0.1, step = 0.1,
                  min_samp = 20):
    '''Get each simulation's SFMS
    
    Inputs:
    - sim (String): name of simulation
    - ax (matplotlib axes): axis to plot data
    - STARS_OR_GAS (String): define gas-phase or stellar metallicity
    - THRESHOLD (float): impacts selection criteria
    - width (float): Width of mass bins
    - step (float): amount to increase mass bins by
    - min_sample (int): minimum number of galaxies in a bin
    
    Returns:
    - (list): The cmasher colors used to make the plot
    '''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,False,THRESHOLD=THRESHOLD)
    
    unique = np.unique(redshifts)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    
    ## Holders for minimum SFR at each redshift
    S_min_bin = []
    m_min = 0
    
    for index, snap in enumerate(snapshots):
        
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS)
        
        MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                       width=width, min_samp=min_samp)
        
        MZR_M_real -= width/2 ## Center bins
        
        ## Append lowest mass bin, if it exists
        if len(real_SFR) > 0:
            S_min_bin.append(np.log10(real_SFR[0]))
        else:
            S_min_bin.append(np.nan)
        
        if index == 0: # if z=0, print the slope
            m_min = MZR_M_real[0]
            a,b = np.polyfit(MZR_M_real, np.log10(real_SFR), 1)
            print(f"SFMS Slope @ z=0: {a:0.3f}")
        
        color = colors[index]
        lw = 2.5
        ax.plot( MZR_M_real, np.log10(real_SFR), color=color,
                      label=r'$z=%s$' %index, lw=lw )

    ## Get difference between z=0 minimum SFR and all others
    S_min_bin_no_subtract = np.array(S_min_bin)
    S_min_bin -= S_min_bin_no_subtract[0]
        
    xstart = 9.6
    ystart = -3
    xlen = 1.8
    ylen = 2.2
        
    ax_inset = ax.inset_axes([xstart, ystart, xlen, ylen], transform=ax.transData)
    ax_inset.scatter(unique, S_min_bin, c=colors, s=10)
        
    ax_inset.spines['bottom'].set_linewidth(1); ax_inset.spines['top'].set_linewidth(1)
    ax_inset.spines['left'].set_linewidth(1)  ; ax_inset.spines['right'].set_linewidth(1)
        
    ax_inset.tick_params(axis='both', which='major', length=3, width=1.5)
    ax_inset.tick_params(axis='both', which='minor', length=2, width=1)
    
    ax_inset.tick_params(axis='y', labelsize=10)
    ax_inset.tick_params(axis='x', which='both', top=True, labelsize=10)
    
    ax_inset.set_ylim(-0.25,2.25)
    ax_inset.set_xlim(-1,9)
    
    ax_inset.text(-0.2,0.5,r'${\rm Offset~(dex)}$', fontsize=10,
                  transform=ax_inset.transAxes, va='center', rotation=90)
    ax_inset.text(0.5,-0.35,r'${\rm Redshift}$', fontsize=10,
                  transform=ax_inset.transAxes, ha='center')
    
    ax_inset.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax_inset.set_xticks([0,4,8])
    ax_inset.set_yticks([0,1,2])
    
    ymin = np.nanmin(S_min_bin_no_subtract)
    ymax = np.nanmax(S_min_bin_no_subtract)
    
    x = np.linspace(m_min, xstart, 100)
    
    slope1 = (ystart - ymin) / (xstart - m_min)
    slope2 = (ystart + ylen - ymax) / (xstart - m_min)
    
    y1 = slope1 * (x - xstart) + ystart
    y2 = slope2 * (x - xstart) + ystart + ylen
    
    ax.plot(x, y1, color='k', alpha=0.5, lw=1)
    ax.plot(x, y2, color='k', alpha=0.5, lw=1)
    ax.vlines(x=m_min, ymin=ymin, ymax=ymax, color='k', lw=1)
        
    return colors


if __name__ == "__main__":
    
    print("Hello World!")

########################################
######### DEPRECATED GRAVEYARD #########
########################################

def get_n_movie_frames(sim,STARS_OR_GAS="gas"):
    '''Deprecated'''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    snap = snapshots[0]
    
    star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS)
        
    width = 0.1
    min_samp = 20
    MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                   width=width,
                                                   min_samp=min_samp)

    return len(MZR_M_real)

def make_FMR_fig(sim,all_z_fit,STARS_OR_GAS="gas",savedir="./",
                 function = fourth_order,verbose=True):
    '''Deprecated'''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,all_z_fit)

    min_alpha, *params = get_z0_alpha(sim, function=function)
    print(min_alpha)
    best_mu = star_mass - min_alpha*np.log10(SFR)
    if (all_z_fit):
        params, cov = curve_fit(function, best_mu, Z_use)

    if verbose:
        print(f'Simulation: {sim}')
        print(f'All_redshift: {all_z_fit}')
        print(f'Params: {params}')

    plot_mu = np.linspace(np.min(best_mu),np.max(best_mu),100)
    best_line = function(plot_mu, *params)

    unique, n_gal = np.unique(redshifts, return_counts=True)

    CMAP_TO_USE = cmr.get_sub_cmap('cmr.guppy_r', 0.0, 1.0, N=len(redshifts))

    plt.clf()
    fig = plt.figure(figsize=(30,20))

    gs  = gridspec.GridSpec(4, 7, width_ratios = [0.66,0.66,0.66,0.35,1,1,1],
                             height_ratios = [1,1,1,0.4], wspace = 0.0, hspace=0.0)

    axBig = fig.add_subplot( gs[:,:3] )

    Hist1, xedges, yedges = np.histogram2d(best_mu,Z_use,weights=redshifts,bins=(100,100))
    Hist2, _     , _      = np.histogram2d(best_mu,Z_use,bins=[xedges,yedges])

    Hist1 = np.transpose(Hist1)
    Hist2 = np.transpose(Hist2)

    hist = Hist1/Hist2

    mappable = axBig.pcolormesh( 
        xedges, yedges, hist, vmin = 0, vmax = 8, cmap=CMAP_TO_USE, rasterized=True
    )

    cbar = plt.colorbar( mappable, label=r"${\rm Redshift}$", orientation='horizontal' )
    cbar.ax.set_xticks(np.arange(0,9))
    cbar.ax.set_xticklabels(np.arange(0,9))

    axBig.plot( plot_mu, best_line, color='k', lw=8.0 )

    if (STARS_OR_GAS == "GAS"):
        plt.ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$')
    elif (STARS_OR_GAS == "STARS"):
        plt.ylabel(r'$\log(Z_* [Z_\odot])$')
    plt.xlabel(r'$\mu_{%s} = \log M_* - %s\log{\rm SFR}$' %(min_alpha,min_alpha))

    axBig.text( 0.05, 0.9, "%s" %WHICH_SIM_TEX[sim], transform=axBig.transAxes )

    Hist1, xedges, yedges = np.histogram2d(best_mu,Z_use,bins=(100,100))
    Hist2, _     , _      = np.histogram2d(best_mu,Z_use,bins=[xedges,yedges])

    percentage = 0.01
    xmin, xmax = np.min(best_mu)*(1-percentage), np.max(best_mu)*(1+percentage)
    ymin, ymax = np.min(Z_use)  *(1-percentage), np.max(Z_use)  *(1+percentage)

    axBig.set_xlim(xmin,xmax)
    axBig.set_ylim(ymin,ymax)

    Hist1 = np.transpose(Hist1)
    Hist2 = np.transpose(Hist2)

    hist = Hist1/Hist2

    ax_x = 0
    ax_y = 4

    axInvis = fig.add_subplot( gs[:,3] )
    axInvis.set_visible(False)

    small_axs = []
    ylab_flag = True

    for index, time in enumerate(unique):
        ax = fig.add_subplot( gs[ax_x, ax_y] )

        small_axs.append(ax)

        if (ylab_flag):
            if (STARS_OR_GAS == "GAS"):
                ax.set_ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$',fontsize=36 )
            elif (STARS_OR_GAS == "STARS"):
                ax.set_ylabel(r'$\log(Z_* [Z_\odot])$',fontsize=36 )

            ylab_flag = False

        if (ax_x == 2):
            ax.set_xlabel( r'$\mu_{%s}$' %(min_alpha),
                           fontsize=36 )

        if (ax_y == 5 or ax_y == 6):
            ax.set_yticklabels([])
        if (ax_y == 0 or ax_y == 1):
            ax.set_xticklabels([])

        ax_y += 1
        if (ax_y == 7):
            ax_y = 4
            ax_x += 1
            ylab_flag = True

        mask = (redshifts == time)

        ax.pcolormesh( xedges, yedges, hist, alpha=0.25, vmin = 0, vmax = 1.5,
                       cmap=plt.cm.Greys, rasterized=True )

        current_mu    =   best_mu[mask]
        current_Z     =     Z_use[mask]

        Hist1, current_x, current_y = np.histogram2d(current_mu,current_Z,bins=[xedges, yedges])
        Hist2, _        , _         = np.histogram2d(current_mu,current_Z,bins=[current_x,current_y])

        Hist1 = np.transpose(Hist1)
        Hist2 = np.transpose(Hist2)

        current_hist = Hist1/Hist2

        vmin = 1 - time
        vmax = 9 - time

        ax.pcolormesh( 
            current_x, current_y, current_hist, vmin = vmin, vmax = vmax,
            cmap=CMAP_TO_USE, rasterized=True 
        )

        ax.plot( plot_mu, best_line, color='k', lw=6 )

        ax.text( 0.65, 0.1, r"$z = %s$" %int(time), transform=plt.gca().transAxes )

        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    if function == linear:
        _a_, _b_ = params
        x=9.25
        l2 = np.array((x, _a_*x+_b_+0.05))
        rotation = np.arctan(_a_)
        rotation = np.degrees(rotation)
        rotation = axBig.transData.transform_angles(np.array((rotation,)),
                                                    l2.reshape((1, 2)))[0]
        text = ''
        if all_z_fit:
            text = r'${\rm All}~z~{\rm Fit~FMR}$'
        else:
            text =  r'$z=0~{\rm Fit~FMR}$'
        axBig.text( l2[0],l2[1], text, rotation = rotation, rotation_mode='anchor', fontsize=40 )

    plt.tight_layout()   
        
    save_str = "Figure4" if all_z_fit else "Figure1"
    save_str += ("_" + sim.lower()) if sim != "EAGLE" else ""
    save_str += ".pdf"
    
    plt.savefig( savedir + save_str, bbox_inches='tight' )
    plt.clf()
