'''
This module defines several useful functions for Appendix B
#
Functions:
    - get_all_redshifts(sim,all_z_fit,STARS_OR_GAS='gas',
                        THRESHOLD=-5.00E-01)
        Get data from a simulation across all snapshots z=0-8
        
    - get_medians(x,y,z,width=0.05,min_samp=15)
        Get the medians metallicity within fixed mass bins
#
Code written by: Alex Garcia, 2024
'''
### Standard imports
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.gridspec as gridspec
import cmasher as cmr
import h5py
### Scipy
from scipy.optimize import curve_fit
from scipy import stats

### Define matplotlib params
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.linewidth'] = 2.25*1.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5*1.25
mpl.rcParams['ytick.major.width'] = 1.5*1.25
mpl.rcParams['xtick.minor.width'] = 1.0*1.25
mpl.rcParams['ytick.minor.width'] = 1.0*1.25
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4.5
mpl.rcParams['ytick.minor.size'] = 4.5
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

### Where reduced data lives
BLUE = './Data/'

### Constants
h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02
Zsun   = 1.27E-02

### Stellar/Gas Mass Parameters (note this changes for SIMBA)
m_star_min = 8.0
m_star_max = 12.0
m_gas_min  = 8.5

### Helper to convert from simulation name to TeX output
WHICH_SIM_TEX = {
    "TNG":r"${\rm TNG}$",
    "ORIGINAL":r"${\rm Illustris}$",
    "EAGLE":r"${\rm EAGLE}$",
    "SIMBA":r"${\rm SIMBA}$"
}

def get_all_redshifts(sim, metal_type='Subfind'):
    sim = sim.upper()
    
    sim_to_file = {
        "ORIGINAL":"Illustris_metals.hdf5", 
        "TNG":"IllustrisTNG_metals.hdf5",
        "EAGLE":"Eagle_metals.hdf5",
        "SIMBA":"Simba_metals.hdf5"
    }

    masses = []
    metallicity = []
    redshift = []
    
    with h5py.File('./Data/' + sim_to_file[sim], 'r') as f:
        max_z = 9 if sim != "SIMBA" else 8
        zs = np.arange(0, max_z)
        
        # EAGLE z=8 => 8.1
        # SIMBA z=6 => 5.9
        # SIMBA z=7 => 7.1
        
        for z in zs:
            if sim == "SIMBA" and z == 6:
                z = 5.9
            elif sim == "SIMBA" and z == 7:
                z = 7.1
            elif sim == "EAGLE" and z == 8:
                z = 8.1
                
            current_group = f['z=%s' %float(round(z,2))]
            masses += list(current_group['Masses'])
            metallicity += list(current_group[metal_type])
            redshift += list( np.ones(len(current_group['Masses'])) * int(z+0.2) )
            
    return np.array(masses), np.array(metallicity), np.array(redshift)

def get_medians(x,y,z,width=0.05,min_samp=15):
    '''Get the medians metallicity within fixed mass bins
    
    Inputs:
    - x (ndarray): masses
    - y (ndarray): metallicities
    - width (float): mass bin width
    - min_sample (int): minimum number of galaxies in a bin
    
    Returns:
    - (ndarray): median mass bins
    - (ndarray): corresponding metallicity bins
    - (ndarray): corresponding SFR bins
    '''
    start = np.min(x)
    end   = np.max(x)
    
    xs = np.arange(start,end,width)
    median_y = np.zeros( len(xs) )
    median_z = np.zeros( len(xs) )
    
    scatter_y = np.zeros(len(xs))
    
    for index, current in enumerate(xs):
        mask = ((x > (current)) & (x < (current + width)))
        
        if (len(y[mask]) > min_samp):
            median_y [index] = np.nanmedian(y[mask])
            scatter_y[index] = np.nanstd(y[mask])
            median_z [index] = np.nanmedian(z[mask])
        else:
            median_y [index] = np.nan
            scatter_y[index] = np.nan
            median_z [index] = np.nan
        
    nonans = ~(np.isnan(median_y)) & ~(np.isnan(median_z)) & ~(np.isnan(scatter_y))
    
    xs = xs[nonans] + width
    median_y  = median_y [nonans]
    median_z  = median_z [nonans]
    scatter_y = scatter_y[nonans]

    return xs, median_y, median_z, scatter_y

if __name__ == "__main__":
    
    print('Hello World!')