'''
This file is used to create Figure 6 of "Does the fundamental 
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
import cmasher as cmr
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
### From this library
from helpers import (
    WHICH_SIM_TEX, get_z0_alpha, get_medians
)
from plotting import (
    make_MZR_prediction_fig_epsilon,
    linear, fourth_order, format_func
)
mpl.rcParams['axes.linewidth'] = 3.5
mpl.rcParams['xtick.major.width'] = 2.75
mpl.rcParams['ytick.major.width'] = 2.75
mpl.rcParams['xtick.minor.width'] = 1.75
mpl.rcParams['ytick.minor.width'] = 1.75
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['ytick.minor.size'] = 5


sims = ['ORIGINAL','TNG','EAGLE','SIMBA']

redshifts = np.arange(0,9)

function = linear

cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(redshifts))
newcolors = np.linspace(0, 1, len(redshifts))
colors = [ cmap(x) for x in newcolors[::-1] ]

fig, axs_all = plt.subplots(4,4,figsize=(11,13),
                            gridspec_kw={'width_ratios': [1, 1, 0.4, 1]},
                            sharex=True)

ax_column1 = []
ax_column2 = []
ax_column3 = []

YMIN, YMAX = 0,0

for index, sim in enumerate(sims):
    axs = axs_all[index,:]
    ax_real = axs[0]
    ax_column1.append(ax_real)
    ax_fake = axs[1]
    ax_column2.append(ax_fake)
    ax_blank = axs[2]
    ax_offsets = axs[3]
    ax_column3.append(ax_offsets)

    ax_blank.axis('off')
    
    colors, MSE = make_MZR_prediction_fig_epsilon(sim,ax_real,ax_fake,ax_offsets,
                                                 function=function)

    if index == 0:
        ax_fake.text(0.05,0.85,r'${\rm Varied}~\varepsilon_{\rm evo}$',transform=ax_fake.transAxes, fontsize=18)
    
    if index == 3:
        for ax in axs:
            ax.set_xlabel(r'$\log(M_*~[M_\odot])$')
            
    for ax in axs:
        ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    ymin = min(ax_real.get_ylim()[0], ax_fake.get_ylim()[0])
    ymax = max(ax_real.get_ylim()[1], ax_fake.get_ylim()[1])

    for ax in [ax_real, ax_fake]:
        ax.set_ylim(ymin, ymax)

    ax_real.set_xticks([8,9,10,11])

    ax_real.set_ylabel(r'$\log({\rm O/H}) + 12~{\rm (dex)}$')
    ax_fake.set_yticklabels([])
    ax_offsets.set_ylabel(r'${\rm True} - {\rm Predicted}$')

    ax_real.text(0.05,0.85,WHICH_SIM_TEX[sim.upper()],transform=ax_real.transAxes)

    if index == 0:
        ax_real.text(0.5,1.05,r'${\rm True~MZR}$',transform=ax_real.transAxes,ha='center',fontsize=28)
        ax_fake.text(0.5,1.05,r'${\rm Predicted~MZR}$',transform=ax_fake.transAxes,ha='center',fontsize=28)
        ax_offsets.text(0.5,1.05,r'${\rm Residuals}$',transform=ax_offsets.transAxes,ha='center',fontsize=28)

    ax_offsets.axhline(0.0, color='k', linestyle=':', lw=3)

    if index == 0:
        YMIN, YMAX = ax_real.get_ylim()
        
    ax_fake.set_ylim(YMIN, YMAX)
    ax_real.set_ylim(YMIN, YMAX)

    ax_offsets.set_ylim(-0.39,0.49)

    
    if index == 0:
        leg = ax_offsets.legend(frameon=True,labelspacing=0.05,
                                handletextpad=0, handlelength=0, 
                                markerscale=-1,bbox_to_anchor=(1,1.05),
                                framealpha=1, edgecolor='k',fancybox=False)
        for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)
        for index, text in enumerate(leg.get_texts()):
            text.set_color(colors[index])
        leg.get_frame().set_linewidth(3)

plt.tight_layout()

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig('./Figures (pdfs)/' + 'Figure6.pdf', bbox_inches='tight')