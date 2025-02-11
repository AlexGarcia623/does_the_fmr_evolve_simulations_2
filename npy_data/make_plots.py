import numpy as np
import matplotlib.pyplot as plt

sims = ['EAGLE','ORIGINAL','TNG','SIMBA']
redshifts = np.arange(0,9)

#### SFMS ####
fig, axs = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
axs = axs.flatten()

for index, sim in enumerate(sims):
        ax = axs[index]
        for redshift in redshifts:
                mass = np.load(f'./SFMS/{sim}_z={redshift}_mass.npy')
                sfr = np.load(f'./SFMS/{sim}_z={redshift}_sfr.npy')

                ax.plot(mass, np.log10(sfr))
        ax.text(0.05, 0.95, sim, transform=ax.transAxes)

for ax in [axs[0],axs[2]]:
	ax.set_ylabel('log(SFR [Msun/yr])')
for ax in [axs[2], axs[3]]:
	ax.set_xlabel('log(Mstar [Msun])')

plt.tight_layout()
plt.subplots_adjust(hspace=0.0, wspace=0.0)
plt.savefig('SFMS.pdf')

#### MZR ####
fig, axs = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
axs = axs.flatten()

for index, sim in enumerate(sims):
	ax = axs[index]
	for redshift in redshifts:
		mass = np.load(f'./MZR/{sim}_z={redshift}_mass.npy')
		metals = np.load(f'./MZR/{sim}_z={redshift}_metallicity.npy')

		ax.plot(mass, metals)
	ax.text(0.05, 0.95, sim, transform=ax.transAxes)

for ax in [axs[0],axs[2]]:
        ax.set_ylabel('log(O/H) + 12 [dex]')
for ax in [axs[2], axs[3]]:
        ax.set_xlabel('log(Mstar [Msun])')

plt.tight_layout()
plt.subplots_adjust(hspace=0.0, wspace=0.0)
plt.savefig('MZR.pdf')
