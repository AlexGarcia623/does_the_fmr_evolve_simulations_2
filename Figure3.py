'''
This file is used to create Figure 3 of "Does the fundamental 
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
from plotting import (
    make_MZR_fig
)
from helpers import (
    WHICH_SIM_TEX
)
mpl.rcParams['font.size'] = 16


sims = ['original','tng','eagle','simba']

savedir = './Figures (pdfs)/'
STARS_OR_GAS = "gas".upper()

### Initialize Figure
fig, axs_MZR = plt.subplots(2, 2, figsize = (8,5.75), sharex=True, sharey=True)

axs = axs_MZR.flatten()

for index, ax in enumerate(axs):
    sim = sims[index].upper()
    
    colors = make_MZR_fig(sim, ax)
    
    ax.text( 0.05, 0.825, WHICH_SIM_TEX[sim],
             transform=ax.transAxes )

ymin, ymax = axs[0].get_ylim()
axs[0].set_ylim(6.6, ymax*1.03)
xmin, _ = axs[0].get_xlim()
axs[0].set_xlim(xmin, 12.1)
axs[0].set_xticks([8,9,10,11])
    
leg = axs[1].legend(frameon=True,labelspacing=0.05,
                    handletextpad=0, handlelength=0, 
                    markerscale=-1,bbox_to_anchor=(1.275,1.05),
                    framealpha=1, edgecolor='black',fancybox=False)
for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)
for index, text in enumerate(leg.get_texts()):
    text.set_color(colors[index])
leg.get_frame().set_linewidth(2.5)

axs[0].text(-0.1,0,r'$\log ({\rm O/H}) + 12 ~({\rm dex})$',
            ha='center', va='center', rotation=90, transform=axs[0].transAxes)
axs[2].text(1.0,-0.175,r'$\log (M_*~[M_\odot])$', ha='center',
            transform=axs[2].transAxes)

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(savedir + 'Figure3.pdf', bbox_inches='tight')