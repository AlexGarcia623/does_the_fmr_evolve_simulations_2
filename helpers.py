'''
This module defines several useful functions for the analysis
of this work
#
Functions:
    - switch_sim(WHICH_SIM)
        Get constants in order to switch which simulation to analyze
        
    - fourth_order(mu, a, b, c, d, e)
        Creates fourth order regression
        
    - third_order(mu, a, b, c, d)
        Creates third order regression
        
    - linear(mu, a, b)
        Creates linear regression
        
    - Curti_MZR(mass,Z0,gamma,beta,M0)
        Creates Regression based on MZR from Curti+2020
        
    - Curti_FMR(X,Z0,gamma,beta,m0,m1)
        Creates regression based on FMR from Curti+2020
        
    - get_all_redshifts(sim,all_z_fit,STARS_OR_GAS='gas',
                        THRESHOLD=-5.00E-01)
        Get data from a simulation across all snapshots z=0-8
        
    - get_one_redshift(BLUE_DIR,snap,STARS_OR_GAS='gas',
                     THRESHOLD=-5.00E-01,snap2z=None)
        Get data from a simulation at one snapshot
        
    - get_z0_alpha(sim,STARS_OR_GAS='gas',function=None)
        Get the projection of minimum scatter at z=0
        
    - linear_mu(mu, a, b)
        Essentially just a linear function... too scared to delete
        
    - fourth_order_mu(mu, a, b, c, d, e)
        Essentially just a fourth order function... too scared to delete
        
    - ttest(hypothesized_value,measurements)
        Perform 1 sample t-test
        
    - ttest_w_errs(hypothesized_value,measurements,errors) 
        Perform 1 sample t-test with errors
        
    -  estimate_symmetric_error(lower, upper)
        Some Errors are non symmetric, but not by much I am
        just estimating them here
        
    - sfmscut(m0, sfr0, THRESHOLD=-5.00E-01,m_star_min=8.0)
        Compute specific star formation main sequence.
        Adapted from Hemler+(2021)
        
    - get_medians(x,y,z,width=0.05,min_samp=15)
        Get the medians metallicity within fixed mass bins
        
    - get_alpha_min(sim, m_star_min, m_star_max, m_gas_min=8.5, STARS_OR_GAS='gas',
                  polyorder=1,THRESHOLD=-5.00E-01,verbose=False)
        Get projection of minimum scatter at all redshifts
        Used for Appendix C. Directly adapted from Garcia+(2024b)
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
zo     = 5.000E-01
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

def switch_sim(WHICH_SIM):
    '''Get constants in order to switch which simulation to analyze
    
    Inputs:
    - WHICH_SIM: any of (eagle, original, TNG, simba)
    
    Returns
    - (list): snapshots
    - (dict): snap2z
    - (String): location in Data
    '''
    BLUE_DIR = BLUE + WHICH_SIM + "/"
    if (WHICH_SIM.upper() == "TNG"):
        # TNG
        run       = 'L75n1820TNG'
        base      = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/' 
        out_dir   = base 
        snapshots = [99,50,33,25,21,17,13,11,8] # 6,4
        snap2z = {
            99:'z=0',
            50:'z=1',
            33:'z=2',
            25:'z=3',
            21:'z=4',
            17:'z=5',
            13:'z=6',
            11:'z=7',
            8 :'z=8',
            6 :'z=9',
            4 :'z=10',
        }
    elif (WHICH_SIM.upper() == "ORIGINAL"):
        # Illustris
        run       = 'L75n1820FP'
        base      = '/orange/paul.torrey/Illustris/Runs/' + run + '/'
        out_dir   = base
        snapshots = [135,86,68,60,54,49,45,41,38] # 35,32
        snap2z = {
            135:'z=0',
            86 :'z=1',
            68 :'z=2',
            60 :'z=3',
            54 :'z=4',
            49 :'z=5',
            45 :'z=6',
            41 :'z=7',
            38 :'z=8',
            35 :'z=9',
            32 :'z=10',
        }
    elif (WHICH_SIM.upper() == "EAGLE"):
        snapshots = [28,19,15,12,10,8,6,5,4] # 3,2
        snap2z = {
            28:'z=0',
            19:'z=1',
            15:'z=2',
            12:'z=3',
            10:'z=4',
             8:'z=5',
             6:'z=6',
             5:'z=7',
             4:'z=8',
             3:'z=9',
             2:'z=10'
        }
    elif (WHICH_SIM.upper() == "SIMBA"):
        snapshots = [151, 105, 79, 62, 51, 42, 36, 30, 26]
        snap2z = {
            151:'z=0',
            105:'z=1',
             79:'z=2',
             62:'z=3',
             51:'z=4',
             42:'z=5',
             36:'z=6',
             30:'z=7',
             26:'z=8'
        }
    return snapshots, snap2z, BLUE_DIR


def fourth_order(mu, a, b, c, d, e):
    '''Creates fourth order regression
    
    Inputs:
    - mu (ndarray): x-axis data
    - a (float): fit parameter 1
    - b (float): fit parameter 2
    - c (float): fit parameter 3
    - d (float): fit parameter 4
    - e (float): fit parameter 5

    Returns:
    - (ndarray): a*mu**4 + b*mu**3 + c*mu**2 + d*mu + e
    '''
    return a * mu**4 + b * mu**3 + c * mu**2 + d * mu + e

def third_order(mu, a, b, c, d):
    '''Creates third order regression
    
    Inputs:
    - mu (ndarray): x-axis data
    - a (float): fit parameter 1
    - b (float): fit parameter 2
    - c (float): fit parameter 3
    - d (float): fit parameter 4

    Returns:
    - (ndarray): a*mu**3 + b*mu**2 + c*mu + d
    '''
    return a * mu**3 + b * mu **2 + c * mu + d

def linear(mu, a, b):
    '''Creates linear regression
    
    Inputs:
    - mu (ndarray): x-axis data
    - a (float): slope
    - b (float): intercept

    Returns:
    - (ndarray): a * mu + b
    '''
    return a * mu + b

def Curti_MZR(mass,Z0,gamma,beta,M0):
    '''Creates regression based on MZR from Curti+2020
    
    Inputs:
    - mass (ndarray): x-axis data
    - Z0 (float): fit parameter 1
    - gamma (float): fit parameter 2
    - beta (float): fit parameter 3
    - M0 (float): fit parameter 4
    
    Returns:
    - (ndarray): Z0 - gamma/beta * np.log10(1 + (mass/M0)**(-beta))
    '''
    return Z0 - gamma/beta * np.log10(1 + (mass/M0)**(-beta))

def Curti_FMR(X,Z0,gamma,beta,m0,m1):
    '''Creates regression based on FMR from Curti+2020
    
    Inputs:
    - X (list): Contains mass array as 0 index, SFR array in 1 index
    - Z0 (float): fit parameter 1
    - gamma (float): fit parameter 2
    - beta (float): fit parameter 3
    - m0 (float): fit parameter 4
    - m1 (float): fit parameter 5
    
    Returns:
    - (ndarray): Z0 - gamma/beta * np.log10(1 + (X[0]/(10**m0 * X[1] **m1))**(-beta))
    '''
    mass = X[0]
    SFR  = X[1]
    M0 = 10**m0 * SFR **m1
    return Z0 - gamma/beta * np.log10(1 + (mass/M0)**(-beta) )

def get_all_redshifts(sim,all_z_fit,STARS_OR_GAS='gas',THRESHOLD=-5.00E-01):
    '''Get data from a simulation across all snapshots z=0-8
    
    Inputs:
    - WHICH_SIM (String): any of (eagle, original, TNG, simba)
    - all_z_fit (boolean): DEPRECATED
    - STARS_OR_GAS (String): define gas-phase or stellar metallicity
    - THRESHOLD (float): impacts selection criteria
    
    Returns
    - (ndarray): stellar masses
    - (ndarray): SFRs
    - (ndarray): metallicity
    - (ndarray): redshifts
    '''
    all_z_fit = False ## deprecated 
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    ## Stellar/Gas Mass Limits
    m_star_min = 8.0
    m_star_max = 12.0
    m_gas_min  = 8.5
    if sim == "SIMBA": ## Simba is lower res -- higher minimum thresholds
        m_star_min = 9.0 
        m_gas_min = 9.5
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    redshifts  = []
    redshift   = 0
    
    for snap in snapshots:
        currentDir = BLUE_DIR + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )
        
        sfms_idx = sfmscut(star_mass, SFR, THRESHOLD, m_star_min)
        if sim != "SIMBA": ## SIMBA saved differently
            desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                            (star_mass < 1.00E+01**(m_star_max)) &
                            (gas_mass  > 1.00E+01**(m_gas_min))  &
                            (sfms_idx))
        
            gas_mass  =  gas_mass[desired_mask]
            star_mass = star_mass[desired_mask]
            SFR       =       SFR[desired_mask]
            Zstar     =     Zstar[desired_mask]
            Zgas      =      Zgas[desired_mask]
            R_gas     =     R_gas[desired_mask]
            R_star    =    R_star[desired_mask]
        
        all_Zgas     += list(Zgas     )
        all_Zstar    += list(Zstar    )
        all_star_mass+= list(star_mass)
        all_gas_mass += list(gas_mass )
        all_SFR      += list(SFR      )
        all_R_gas    += list(R_gas    )
        all_R_star   += list(R_star   )

        redshifts += list( np.ones(len(Zgas)) * redshift )
        
        redshift  += 1
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )
    redshifts = np.array(redshifts     )

    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    Zgas      = Zgas     [nonans]
    redshifts = redshifts[nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)
    
    Z_use = None
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    return star_mass, SFR, Z_use, redshifts
    

def get_one_redshift(BLUE_DIR,snap,STARS_OR_GAS='gas',
                     THRESHOLD=-5.00E-01,snap2z=None):
    '''Get data from a simulation at one snapshot
    
    Inputs:
    - WHICH_SIM (String): any of (eagle, original, TNG, simba)
    - snap (int): snapshot of interest
    - STARS_OR_GAS (String): define gas-phase or stellar metallicity
    - THRESHOLD (float): impacts selection criteria
    - snap2z (dict): converts snapshot number to redshift
    
    Returns
    - (ndarray): stellar masses
    - (ndarray): metallicity
    - (ndarray): SFRs
    '''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    
    currentDir = BLUE_DIR + 'snap%s/' %snap

    Zgas      = np.load( currentDir + 'Zgas.npy' )
    Zstar     = np.load( currentDir + 'Zstar.npy' ) 
    star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
    gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
    SFR       = np.load( currentDir + 'SFR.npy' )
    R_gas     = np.load( currentDir + 'R_gas.npy' )
    R_star    = np.load( currentDir + 'R_star.npy' )

    ## Stellar/Gas Mass Limits
    m_star_min = 8.0
    m_star_max = 12.0
    m_gas_min  = 8.5
    if "SIMBA" in BLUE_DIR: ## Simba is lower res -- higher minimum thresholds
        m_star_min = 9.0
        m_gas_min  = 9.5
        
    sfms_idx = sfmscut(star_mass, SFR, m_star_min = m_star_min)
    if "SIMBA" not in BLUE_DIR:
        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))

        gas_mass  =  gas_mass[desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       =       SFR[desired_mask]
        Zgas      =      Zgas[desired_mask]
    
    OH     = Zgas * (zo/xh) * (1.00/16.00)
    Zgas   = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass

    sSFR[~(sSFR > 0.0)] = 1e-16

    star_mass = star_mass[nonans]
    sSFR      = sSFR     [nonans]
    Zgas      = Zgas     [nonans]

    star_mass     = np.log10(star_mass)

    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar
        
    return star_mass, Z_use, SFR
    
def get_z0_alpha(sim,STARS_OR_GAS='gas',function=None):
    '''Get the projection of minimum scatter at z=0
    
    Inputs:
    - sim (String): name of simulation (eagle, original, tng)
    - STARS_OR_GAS (String): Get stellar or gas-phase metallicity
    - function (python function): function to calculate scatter about
    
    Returns:
    - (float): alpha_min
    - (float): all parameters (note that this returns an unpacked list)
    '''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    print('Getting z=0 alpha: %s' %sim)
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    snap = snapshots[0]
    
    m_star_min = 8.0
    m_star_max = 12.0
    m_gas_min = 8.5
    
    if sim == "SIMBA":
        m_star_min = 9.0
        m_gas_min = 9.5

    currentDir = BLUE_DIR + 'snap%s/' %snap

    Zgas      = np.load( currentDir + 'Zgas.npy' )
    Zstar     = np.load( currentDir + 'Zstar.npy' ) 
    star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
    gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
    SFR       = np.load( currentDir + 'SFR.npy' )
    R_gas     = np.load( currentDir + 'R_gas.npy' )
    R_star    = np.load( currentDir + 'R_star.npy' )

    
    sfms_idx = sfmscut(star_mass, SFR, m_star_min = m_star_min)
    if sim != "SIMBA":
        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))

        gas_mass  =  gas_mass[desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       =       SFR[desired_mask]
        Zstar     =     Zstar[desired_mask]
        Zgas      =      Zgas[desired_mask]
        R_gas     =     R_gas[desired_mask]
        R_star    =    R_star[desired_mask]

    all_Zgas     += list(Zgas     )
    all_Zstar    += list(Zstar    )
    all_star_mass+= list(star_mass)
    all_gas_mass += list(gas_mass )
    all_SFR      += list(SFR      )
    all_R_gas    += list(R_gas    )
    all_R_star   += list(R_star   )
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    # gas_mass  = gas_mass [nonans]
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    # Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    # R_gas     = R_gas    [nonans]
    # R_star    = R_star   [nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)

    alphas = np.linspace(0,1,100)
    all_params = []

    disps = np.ones(len(alphas)) * np.nan
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    ## If using non-linear projection subtract 10 (like Mannucci+2010)
    m10 = 0 
    if function != linear:
        m10 = 10

    for index, alpha in enumerate(alphas):

        mu_fit  = star_mass - alpha*np.log10( SFR ) 

        Z_fit  =  Z_use

        params, cov = curve_fit(function,mu_fit,Z_fit)
        all_params.append(params)
        interp = function(mu_fit, *params)

        disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 

    argmin = np.argmin(disps)

    params = all_params[argmin]
    min_alpha = round( alphas[argmin], 2 )
    
    ## Print Mannucci+2010 parameters
    if m10 == 10:
        mu_fit  = star_mass - min_alpha*np.log10( SFR ) - m10

        Z_fit  =  Z_use

        m10params, cov = curve_fit(function,mu_fit,Z_fit)

        __one__ = '[log ({rm O/H}) + 12]_{%s} = ' %sim
        __two__ = ''
        n_params = len(m10params)
        for index, p in enumerate(m10params):
            power = index
            __two__ += f"{p:0.3f}"
            if power != 0:
                __two__ += f"y^{power}"
            if power != len(m10params)-1:
                __two__ += "+"
                
        print(__one__ + __two__)
            
    return min_alpha, *params

def linear_mu(mu, a, b):
    '''Essentially just a linear function... too scared to delete
    
    Inputs:
    - mu (ndarray): x-data
    - a (float): slope
    - b (float): intercept
    
    Returns:
    - (ndarray): a * mu + b
    '''
    return a * mu + b
        
def fourth_order_mu(mu, a, b, c, d, e):
    '''Essentially just a fourth order function... too scared to delete
    
    Inputs:
    - mu (ndarray): x-data
    - a (float): fit parameter 1
    - b (float): fit parameter 2
    - c (float): fit parameter 3
    - d (float): fit parameter 4
    
    Outputs:
    - (ndarray): a*mu**4 + b*mu**3 + c*mu**2 + d*mu + e
    '''
    return a * mu**4 + b * mu**3 + c * mu**2 + d * mu + e

def ttest(hypothesized_value,measurements):
    '''Perform 1 sample t-test

    Input:
    - hypothesized_value (float): z=0 alpha value to compare against
    - measuements (ndarry): array of alpha values
    '''
    mean = np.mean(measurements)
    res = stats.ttest_1samp(measurements, popmean=hypothesized_value)
    ci = res.confidence_interval(confidence_level=0.95)
    
    t_stat = res.statistic
    p_val = res.pvalue
        
    l=20
    print(f"\t{'Mean':<{l}}: {mean:0.3f}")
    print(f"\t{'Reference':<{l}}: {hypothesized_value:0.3f}")
    print("\t\tStatistical Test")
    print(f"\t{'T-statistic':<{l}}: {t_stat:0.3f}")
    print(f"\t{'P-value':<{l}}: {p_val:0.3E}")
    print(f"\t{'Reject (0.05 level)':<{l}}: {p_val < 0.05}")
    
def ttest_w_errs(hypothesized_value,measurements,errors):
    '''Perform 1 sample t-test with errors

    Input:
    - hypothesized_value (float): z=0 alpha value to compare against
    - measuements (ndarry): array of alpha values
    - errors (ndarray): uncertainty on alpha values
    '''
    l = 20
    # Calculate weighted mean and standard error
    weighted_mean = np.sum(measurements / errors**2) / np.sum(1 / errors**2)
    weighted_std_error = np.sqrt(1 / np.sum(1 / errors**2))

    # Calculate t-statistic
    t_stat = (weighted_mean - hypothesized_value) / weighted_std_error

    # Degrees of freedom
    degrees_freedom = len(measurements) - 1

    # Calculate p-value (two-tailed)
    p_val = 2 * stats.t.sf(np.abs(t_stat), degrees_freedom)

    print(f"\t{'Weighted Mean':<{l}}: {weighted_mean:0.3f}")
    print(f"\t{'ref val':<{l}}: {hypothesized_value:0.3f}")
    print("\t\tStatistical Test")
    print(f"\t{'T-statistic':<{l}}: {t_stat:0.3f}")
    print(f"\t{'P-value':<{l}}: {p_val:0.3E}")
    print(f"\t{'Reject (0.05 level)':<{l}}: {p_val < 0.05}")
    
def estimate_symmetric_error(lower, upper):
    '''Errors are non symmetric, but not by much.
    I am just estimating them here
    
    Inputs:
    - lower (ndarray): all lower bound uncertainties
    - upper (ndarray): all upper bound uncertainties

    Returns:
    - (ndarray): average of uncertainties
    '''
    return (lower + upper) / 2
    
def line(data, a, b):
    '''Repeat function... too scared to delete'''
    return a*data + b

def fourth_order( x, a, b, c, d, e ):
    '''Repeat function... too scared to delete'''
    return a + b*x + c*x**2 + d*x**3 + e*x**4

def third_order( x, a, b, c, d ):
    '''Repeat function... too scared to delete'''
    return a + b*x + c*x**2 + d*x**3

def sfmscut(m0, sfr0, THRESHOLD=-5.00E-01,m_star_min=8.0):
    '''Compute specific star formation main sequence
    
    Adapted from Z.S.Hemler+(2021)
    
    Inputs:
    - m0 (ndarray): mass array
    - sfr0 (ndarray): SFR array
    - THRESHOLD (float): value below which galaxies omitted
    - m_star_min (float): minimum stellar mass
    - m_star_max (float): maximum stellar mass
    
    Returns:
    - (ndarray): boolean array of systems that meet criteria
    '''
    nsubs = len(m0)
    idx0  = np.arange(0, nsubs)
    non0  = ((m0   > 0.000E+00) & 
             (sfr0 > 0.000E+00) )
    m     =    m0[non0]
    sfr   =  sfr0[non0]
    idx0  =  idx0[non0]
    ssfr  = np.log10(sfr/m)
    sfr   = np.log10(sfr)
    m     = np.log10(  m)

    idxbs   = np.ones(len(m), dtype = int) * -1
    cnt     = 0
    mbrk    = 1.0200E+01
    mstp    = 5.0000E-02
    mmin    = m_star_min
    mbins   = np.arange(mmin, mbrk + mstp, mstp)
    rdgs    = []
    rdgstds = []


    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        rdg   = np.median(ssfrb)
        idxb  = (ssfrb - rdg) > THRESHOLD
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
        rdgs.append(rdg)
        rdgstds.append(np.std(ssfrb))

    rdgs       = np.array(rdgs)
    rdgstds    = np.array(rdgstds)
    mcs        = mbins[:-1] + mstp / 2.000E+00
    
    nonans = (~(np.isnan(mcs)) &
              ~(np.isnan(rdgs)) &
              ~(np.isnan(rdgs)))
        
    parms, cov = curve_fit(line, mcs[nonans], rdgs[nonans], sigma = rdgstds[nonans])
    mmin    = mbrk
    mmax    = m_star_max
    mbins   = np.arange(mmin, mmax + mstp, mstp)
    mcs     = mbins[:-1] + mstp / 2.000E+00
    ssfrlin = line(mcs, parms[0], parms[1])
        
    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        idxb  = (ssfrb - ssfrlin[i]) > THRESHOLD
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool        

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
    
    for index, current in enumerate(xs):
        mask = ((x > (current)) & (x < (current + width)))
        
        if (len(y[mask]) > min_samp):
            median_y[index] = np.median(y[mask])
            median_z[index] = np.median(z[mask])
        else:
            median_y[index] = np.nan
            median_z[index] = np.nan
        
    nonans = ~(np.isnan(median_y)) & ~(np.isnan(median_z))
    
    xs = xs[nonans] + width
    median_y = median_y[nonans]
    median_z = median_z[nonans]

    return xs, median_y, median_z

def get_alpha_min(sim, m_star_min, m_star_max, m_gas_min=8.5, STARS_OR_GAS='gas',
                  polyorder=1,THRESHOLD=-5.00E-01,verbose=False):
    '''Used for Appendix C. Directly adapted from Garcia+(2024b)
    https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.1398G/abstract
    
    Get the projection of minimum scatter
    
    Inputs:
    - sim (String): name of simulation (eagle, original, tng)
    - m_star_min (float): minimum stellar mass 
    - m_star_max (float): maximum stellar mass
    - m_gas_min (float): minimum gas mass
    - STARS_OR_GAS (String): Get stellar or gas-phase metallicity
    - polyorder (int): Order of fitting polynomial
    - THRESHOLD (float): threshold for sSFMS (see appendix of paper)
    - verbose (bool): Flag for printing output
    
    Returns:
    - (ndarray): all alpha values for the simulation
    - (ndarray): all lower estimates for alpha
    - (ndarray): all upper estimates for alpha
    '''
    STARS_OR_GAS = STARS_OR_GAS.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    min_alphas = np.zeros(len(snapshots))
    low_errbar = np.zeros(len(snapshots))
    hi_errbar  = np.zeros(len(snapshots))
    
    for gbl_index, snap in enumerate(snapshots):

        currentDir = BLUE_DIR + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )
        
#         sfms_idx = sfmscut(star_mass, SFR, THRESHOLD=THRESHOLD,
#                            m_star_min=m_star_min)

#         desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
#                         (star_mass < 1.00E+01**(m_star_max)) &
#                         (gas_mass  > 1.00E+01**(m_gas_min))  &
#                         (sfms_idx))

#         gas_mass  = gas_mass [desired_mask]
#         star_mass = star_mass[desired_mask]
#         SFR       = SFR      [desired_mask]
#         Zstar     = Zstar    [desired_mask]
#         Zgas      = Zgas     [desired_mask]
#         R_gas     = R_gas    [desired_mask]
#         R_star    = R_star   [desired_mask]
        
        Zstar /= Zsun
        OH     = Zgas * (zo/xh) * (1.00/16.00)

        Zgas      = np.log10(OH) + 12

        # Get rid of nans and random values -np.inf
        nonans    = ~(np.isnan(Zgas)) & (Zgas > 0.0) 

        sSFR      = SFR/star_mass

        star_mass = star_mass[nonans]
        SFR       = SFR      [nonans]
        sSFR      = sSFR     [nonans]
        Zgas      = Zgas     [nonans]

        star_mass = np.log10(star_mass)

        alphas = np.linspace(0,1,100)        
        disps = np.ones(len(alphas)) * np.nan

        if (STARS_OR_GAS == "GAS"):
            Z_use = Zgas
        elif (STARS_OR_GAS == "STARS"):
            Z_use = Zstar
        else:
            break

        for index, alpha in enumerate(alphas):

            muCurrent  = star_mass - alpha*np.log10( SFR )

            mu_fit = muCurrent
            Z_fit  = Z_use

            popt = np.polyfit(mu_fit, Z_fit, polyorder)
            interp = np.polyval( popt, mu_fit )

            disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 

        argmin = np.argmin(disps)
        min_alpha = alphas[argmin]
        min_disp  = disps[argmin]

        if verbose:
            print('%s: %s, best alpha_%s = %s' %(sim, snap2z[snap], STARS_OR_GAS.lower(), min_alpha))

        width = min_disp * 1.05

        within_uncertainty = alphas[ (disps < width) ]

        min_uncertain = within_uncertainty[0]
        max_uncertain = within_uncertainty[-1] 
        
        min_alphas[gbl_index] = min_alpha
        low_errbar[gbl_index] = min_uncertain
        hi_errbar [gbl_index] = max_uncertain
            
    return min_alphas, low_errbar, hi_errbar

if __name__ == "__main__":
    
    print('Hello World!')