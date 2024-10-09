'''
This file is used to create Figure B2 of "Does the fundamental 
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
from scipy.interpolate import interp1d
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

fig, axs_MZR = plt.subplots(2, 2, figsize = (8.25,5.75), sharex=True, sharey=True)

axs = axs_MZR.flatten()
    
for plot_index, sim in enumerate(sims):
    sim = sim.upper()
    star_mass, metallicity, redshift = get_all_redshifts(sim, metal_type='Subfind_so')
    _, metallicity2, _ = get_all_redshifts(sim, metal_type='SFRonlyAll')
    
    ax = axs[plot_index]
    all_offsets = []
    for redshift_index, z in enumerate(np.unique(redshift)):
        z_mask = (redshift == z) & (~np.isnan(metallicity)) & (~np.isnan(metallicity2))

        real_mass   = star_mass[z_mask]
        scale = 0.5
        real_metal  = metallicity[z_mask] + np.log10(scale/0.35) ## Change from previous
        real_metal2 = metallicity2[z_mask]
        real_sfr    = np.zeros(len(real_mass)) ## don't care about this for now


        MZR_M_real1, MZR_Z_real, *_ = get_medians(real_mass,real_metal,real_sfr,
                                                 width=width,min_samp=20)

        MZR_M_real, MZR_Z2_real, *_ = get_medians(real_mass, real_metal2, real_sfr,
                                                  width=width, min_samp=20)
        
        assert(len(MZR_Z_real) == len(MZR_Z2_real))
        
        MZR_M_real -= width/2 ## Center bins

        color = colors[redshift_index]
        lw = 2.5
        ax.plot( MZR_Z2_real, MZR_Z_real, color=color,
                 label=r'$z=%s$' %redshift_index, lw=lw )
    
    ax.text( 0.05, 0.825, WHICH_SIM_TEX[sim],
                     transform=ax.transAxes )

leg = axs[1].legend(frameon=True,labelspacing=0.05,
                    handletextpad=0, handlelength=0, 
                    markerscale=-1,bbox_to_anchor=(1.0,1.05),
                    framealpha=1, edgecolor='black',fancybox=False)
for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)
for index, text in enumerate(leg.get_texts()):
    text.set_color(colors[index])
leg.get_frame().set_linewidth(2.5)

xmin, xmax = axs[0].get_xlim()
for ax in axs:
    line1, = ax.plot([xmin,xmax],[xmin,xmax],color='k', lw=2, alpha=0.75, zorder=-1)
    line2  = ax.fill_between([xmin,xmax],[xmin-0.1,xmax-0.1],[xmin+0.1,xmax+0.1], color='gray', alpha=0.33)
axs[0].set_xlim(xmin,xmax); axs[0].set_ylim(xmin,xmax)
handles = [line1, line2]
labels  = [r'${\rm Equality~Line}$',r'$\pm0.1~{\rm dex}$']

legend = axs[3].legend(handles=handles, labels=labels, loc='center left',
                       frameon=False, bbox_to_anchor=(0.45,1.15), fontsize=12)

axs[0].text(-0.15,0,r'$\log{(\rm O/H)} + 12~({\rm dex})~[{\rm Scaled~Z}]$',
            ha='center', va='center', rotation=90, transform=axs[0].transAxes)
axs[2].text(1.0,-0.175,r'$\log{(\rm O/H)} + 12~({\rm dex})~[{\rm Direct~O~\&~H}]$', ha='center',
            transform=axs[2].transAxes)
plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(savedir+'AppendixB2.pdf', bbox_inches='tight')