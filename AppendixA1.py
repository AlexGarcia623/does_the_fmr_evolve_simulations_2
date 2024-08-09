'''
This file is used to create Figure A1 of "Does the fundamental 
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
import matplotlib.ticker as ticker
import cmasher as cmr
### From this library
from plotting import (
    make_MZR_prediction_fig, linear
)


savedir = './Figures (pdfs)/'
STARS_OR_GAS = "gas".upper()

sims = ['original','tng','eagle','simba']
sim_vals = [0, 1, 2, 3]

nominal_sSFMS = -5.00E-01
low_sSFMS = -1.00E-01
hi_sSFMS = -1.00E00
no_sSFMS = -np.inf

thres = [nominal_sSFMS, low_sSFMS, hi_sSFMS, no_sSFMS]

cmap = cmr.toxic
num_colors = len(thres)
intervals = np.linspace(0.15, 0.85, num_colors)
colors = [cmap(interval) for interval in intervals]

labels = [r'${\rm sSFMS} - 0.1~{\rm dex}$',
          r'${\rm sSFMS} - 0.5~{\rm dex~(fiducial)}$',
          r'${\rm sSFMS} - 1.0~{\rm dex}$',
          r'${\rm SFR}>0$']
offset = [-0.05,-0.025,0.025,0.05]

markersize = 60
marker = 'o' 


fig = plt.figure(figsize=(6,4))
ax = plt.gca()

for i, all_z_fit in enumerate([False]): ## Deprecated, just use False
    for j, sim in enumerate(sims):
        for k, threshold in enumerate(thres):
            dummy_fig, dummy_axs = plt.subplots(1,3)

            _, MSE = make_MZR_prediction_fig(sim,False,dummy_axs[0],
                                                  dummy_axs[1], dummy_axs[2],
                                                  function = linear, THRESHOLD=threshold)
            if j == 0 and i == 0:
                ## only do this once
                ax.scatter( sim_vals[j] + offset[k], MSE, color=colors[k], marker=marker,
                            label=labels[k], s=markersize )
            else:
                ax.scatter( sim_vals[j] + offset[k], MSE, color=colors[k], marker=marker,
                            s=markersize)
            plt.close(dummy_fig)

            
leg = ax.legend(frameon=False, loc='upper left',
                handletextpad=0, handlelength=0, 
                markerscale=-1,)
for n, text in enumerate( leg.texts ):
    text.set_color( colors[n] )
for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)

ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.set_xticks(sim_vals)        
ax.set_xticklabels([r'${\rm Illustris}$',r'${\rm TNG}$',r'${\rm EAGLE}$',r'${\rm SIMBA}$'])

ax.set_ylabel(r'${\rm MSE}$')

plt.tight_layout()
plt.savefig(savedir + 'AppendixA1.pdf', bbox_inches='tight')