'''
This file is used to create Figure B1 of "Does the fundamental 
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
### From this library
from helpers_AppendixB import (
    get_all_redshifts, WHICH_SIM_TEX, get_medians
)
mpl.rcParams['font.size'] = 16

savedir = './Figures (pdfs)/'

sims = ["original",'tng','eagle','simba']

width = 0.1

N = 9
cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=N)
newcolors = np.linspace(0, 1, N)
colors = [ cmap(x) for x in newcolors[::-1] ]

fig, axs_MZR = plt.subplots(2, 2, figsize = (8,5.75), sharex=True, sharey=True)

axs = axs_MZR.flatten()

for plot_index, sim in enumerate(sims):
    sim = sim.upper()
    ## Sorry... metallicity => ZO for this script... too lazy to rewrite
    star_mass, metallicity, redshift = get_all_redshifts(sim, metal_type='ZO')
    
    ax = axs[plot_index]
    
    for redshift_index, z in enumerate(np.unique(redshift)):
        z_mask = (redshift == z) & (~np.isnan(metallicity))

        real_mass  = star_mass[z_mask]
        real_metal = metallicity[z_mask]
        real_sfr   = np.zeros(len(real_mass))


        MZR_M_real, MZR_Z_real, real_SFR, scatter_Z = get_medians(real_mass,real_metal,real_sfr,
                                                                  width=width,min_samp=20)
        
        MZR_M_real -= width/2 ## Center bins

        color = colors[redshift_index]
        lw = 2.5
        ax.plot( MZR_M_real, MZR_Z_real, color=color,
                         label=r'$z=%s$' %redshift_index, lw=lw )

        ax.fill_between( MZR_M_real, MZR_Z_real - scatter_Z, MZR_Z_real + scatter_Z, color=color, alpha=0.2 )
        
    star_mass, metallicity, redshift = get_all_redshifts(sim, metal_type='XH')
    
    ax = axs[plot_index]
    
    for redshift_index, z in enumerate(np.unique(redshift)):
        z_mask = (redshift == z) & (~np.isnan(metallicity))

        real_mass  = star_mass[z_mask]
        real_metal = metallicity[z_mask]
        real_sfr   = np.zeros(len(real_mass))


        MZR_M_real, MZR_Z_real, real_SFR, scatter_Z = get_medians(real_mass,real_metal,real_sfr,
                                                                  width=width,min_samp=20)
        
        MZR_M_real -= width/2 ## Center bins

        color = colors[redshift_index]
        lw = 2.5
        ax.plot( MZR_M_real, MZR_Z_real, color=color, lw=lw, ls='--' )

        ax.fill_between( MZR_M_real, MZR_Z_real - scatter_Z, MZR_Z_real + scatter_Z, color=color, alpha=0.2 )
        
    ax.text( 0.05, 0.1, WHICH_SIM_TEX[sim],
                 transform=ax.transAxes )

ymin, ymax = axs[0].get_ylim()
xmin, _ = axs[0].get_xlim()
axs[0].set_xlim(xmin, 12.1)
axs[0].set_xticks([8,9,10,11])
axs[0].set_yticks([0.3,0.4,0.5,0.6,0.7])

leg = axs[1].legend(frameon=True,labelspacing=0.05,
                    handletextpad=0, handlelength=0, 
                    markerscale=-1,bbox_to_anchor=(1.275,1.05),
                    framealpha=1, edgecolor='black',fancybox=False)
for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)
for index, text in enumerate(leg.get_texts()):
    text.set_color(colors[index])
leg.get_frame().set_linewidth(2.5)

axs[0].text(-0.133,0,r'${\rm Ratio}$',
            ha='center', va='center', rotation=90, transform=axs[0].transAxes)
axs[2].text(1.0,-0.175,r'$\log (M_*~[M_\odot])$', ha='center',
            transform=axs[2].transAxes)

xmin, xmax = axs[0].get_xlim()
ymin, ymax = axs[0].get_ylim()
for ax in axs:
    line1, = ax.plot([xmin-1,xmin-1],[xmin-1,xmin-1],color='k', lw=2.5 )
    line2, = ax.plot([xmin-1,xmin-1],[xmin-1,xmin-1],color='k', lw=2.5, ls='--' )
handles = [line2, line1]
labels  = [r'$X_{\rm H}$',r'$f_{\rm O}$']

legend = axs[3].legend(handles=handles, labels=labels, loc='center left',
                       frameon=False, bbox_to_anchor=(0.675,1.15), fontsize=12)

axs[0].set_xlim(xmin, xmax)
axs[0].set_ylim(ymin, ymax)

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(savedir+'AppendixB1.pdf', bbox_inches='tight')
