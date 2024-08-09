'''
This file is used to create Figure 5 of "Does the fundamental 
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
from alpha_types import (
    get_epsilon_evo
)
from helpers import (
    ttest, estimate_symmetric_error, WHICH_SIM_TEX,
    linear, fourth_order
)
mpl.rcParams['font.size'] = 15


savedir='./Figures (pdfs)/'

sims = ['original','tng','eagle','simba']
lCol = ['C1','C2','C0','C3']
ls   = [':','--','-.','-']
markers = ['^','*','o','s']
offsets = [0.00,0.05,-0.05,-0.05]

### Intialize Figure
fig = plt.figure(figsize=(7,3.5))

all_epsilon = []
all_lower   = []
all_upper   = []

for index,sim in enumerate(sims):
    redshifts = np.arange(0,9)
    if sim.upper() == "SIMBA":
        redshifts = np.arange(0,8)
    eps_evo, lower, upper, all_epsilons = get_epsilon_evo(sim, n_bootstrap=1000)
    
    lower = eps_evo - lower
    lower = np.where(lower < 0, 0, lower)
    upper = upper - eps_evo
    upper = np.where(upper < 0, 0, upper)
    
    all_epsilon.append(eps_evo)
    all_lower.append(lower)
    all_upper.append(upper)
    
    color = lCol[index]
    marker = markers[index]
    offset = offsets[index]
    ms = 7
    if marker == '*':
        ms = 10
    
    no_errs = len(lower[1:])
    
    plt.errorbar( redshifts[1:]+offset, eps_evo[1:], label=WHICH_SIM_TEX[sim.upper()],
                  alpha=0.75, yerr = [np.zeros(no_errs), np.zeros(no_errs)], color=color,
                  linestyle='none', marker=marker, markersize=ms)
    
    sample_mean = np.mean(eps_evo[1:])
plt.xlabel(r'${\rm Redshift}$')
plt.ylabel(r'$\varepsilon_{\rm evo}$')

leg  = plt.legend(frameon=True,handletextpad=0, handlelength=0,
                  markerscale=0,labelspacing=0.05)
lCol = ['C1','C2','C0','C3']
for n, text in enumerate( leg.texts ):
    text.set_color( lCol[n] )
    
handles, labels = plt.gca().get_legend_handles_labels()
handles = [h[0] for h in handles]
leg = plt.legend(handles, labels, frameon=True,handletextpad=0.4, handlelength=0,labelspacing=0.01,
                 loc='upper left')
for n, text in enumerate( leg.texts ):
    text.set_color( lCol[n] )

leg.get_frame().set_alpha(1.0)
leg.get_frame().set_edgecolor('white')

xmin, xmax = plt.xlim()
plt.xlim(xmin-1,xmax)
plt.xticks(np.arange(0,9,1))
plt.tight_layout()

plt.savefig( savedir + 'Figure5.pdf',bbox_inches='tight')

### Do one-sample t-test
for index,epss in enumerate(all_epsilon):
    ### For each simulation perform a t-test
    which_redshift_compare = 1
    hypothesized_value = epss[which_redshift_compare]
    
    print(f'{sims[index].upper()}, compared to z={which_redshift_compare} epsilon value')
    ttest(hypothesized_value, epss[1:])