'''
This file is used to create Figure C1 of "Does the fundamental 
metallicity relation evolve with redshift? II: The Evolution in
Normalisation of the Mass-Metallicity Relation"
#
Paper: https://ui.adsabs.harvard.edu/abs/2024arXiv240706254G/abstract
#
Code written by: Alex Garcia, 2024
'''
### Standard Imports
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
### From this library
from helpers import (
    ttest_w_errs, estimate_symmetric_error, get_alpha_min
)
mpl.rcParams['axes.linewidth'] = 3.5
mpl.rcParams['xtick.major.width'] = 2.75
mpl.rcParams['ytick.major.width'] = 2.75
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2


### Everything from here is directly adapted from Garcia+2024b
### See https://github.com/AlexGarcia623/does_the_fmr_evolve_simulations/blob/main/getAlpha.py

sims = ['SIMBA']

m_star_min = 9.0 ### Changed for res of simba
m_star_max = 12.0
m_gas_min  = 9.5 ### Changed for res of simba

polyorder=1

SIMBA, SIMBA_lower, SIMBA_upper = get_alpha_min( 'SIMBA',
                                                 m_star_min=m_star_min,
                                                 m_star_max=m_star_max,
                                                 m_gas_min=m_gas_min,
                                                 polyorder=polyorder )

### We don't include z=8 in this analysis
SIMBA = SIMBA[:-1]
SIMBA_lower = SIMBA_lower[:-1]
SIMBA_upper = SIMBA_upper[:-1]

z = np.arange(0,8)

SIMBA_upper = SIMBA_upper - SIMBA
SIMBA_lower = SIMBA - SIMBA_lower

measurements = SIMBA
errors = estimate_symmetric_error( SIMBA_lower, SIMBA_upper )

weighted_mean = np.sum(measurements / errors**2) / np.sum(1 / errors**2)

fig = plt.figure(figsize=(9,4))


ms = 7

plt.errorbar( z-0.05, SIMBA, label=r'${\rm SIMBA}$',
              alpha=0.75, color='C3', yerr=[SIMBA_lower, SIMBA_upper],
              linestyle='none', marker='s', markersize=ms)

plt.axhline(weighted_mean, color='C3',linestyle='-')

leg  = plt.legend(frameon=False,handletextpad=0, handlelength=0,
                  markerscale=0,loc='upper left',labelspacing=0.05)

leg  = plt.legend(frameon=True,handletextpad=0, handlelength=0,
                  markerscale=0,loc='lower right',labelspacing=0.05)
lCol = ['C3']
for n, text in enumerate( leg.texts ):
    text.set_color( lCol[n] )
    
handles, labels = plt.gca().get_legend_handles_labels()
handles = [h[0] for h in handles]
leg = plt.legend(frameon=True,handletextpad=0.75, handlelength=0,labelspacing=0.01,bbox_to_anchor=(1,1))
for n, text in enumerate( leg.texts ):
    text.set_color( lCol[n] )

leg.get_frame().set_alpha(1.0)
leg.get_frame().set_edgecolor('white')

plt.xlabel(r'${\rm Redshift}$')
plt.ylabel(r'$\alpha_{\rm min}$')

plt.scatter(0.25,0.33 ,color='k',alpha=0.5,marker='s')
plt.text(0.35,0.3,r'${\rm M10}$',fontsize=14,alpha=0.5)#,transform=plt.gca().transAxes)

plt.scatter(0.25,0.66 ,color='k',alpha=0.5,marker='s')
plt.text(0.35,0.63,r'${\rm AM13}$',fontsize=14,alpha=0.5)#,transform=plt.gca().transAxes)

plt.scatter(0.25,0.55 ,color='k',alpha=0.5,marker='s')
plt.text(0.35,0.52,r'${\rm C20}$',fontsize=14,alpha=0.5)#,transform=plt.gca().transAxes)
ymin, _ = plt.ylim()

plt.ylim(ymin,1.)

plt.tight_layout()

plt.savefig('Figures (pdfs)/' + "FigureC1.pdf", bbox_inches='tight')

### Do reference value t-test
all_alpha = [SIMBA]
all_lower = [SIMBA_lower]
all_upper = [SIMBA_upper]

for index,alphas in enumerate(all_alpha):
    lower = all_lower[index]
    upper = all_upper[index]
    
    which_redshift_compare = 0
    hypothesized_value = alphas[which_redshift_compare]
    
    est_error = estimate_symmetric_error( lower, upper )
    
    print(f'{sims[index]}, compared to z={which_redshift_compare} alpha value')
    ttest_w_errs(hypothesized_value, alphas, est_error)
