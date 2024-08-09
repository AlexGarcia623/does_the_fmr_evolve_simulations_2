import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import sys
from os import path, mkdir

import h5py

### SIMBA data publicly available: 
###       http://simba.roe.ac.uk/

#########################################################################
################# YOU WILL PROBABLY HAVE TO CHANGE THIS #################
savedir = '../Data/SIMBA/'
#########################################################################

if not (path.exists(savedir)):
    mkdir( savedir )
    print('New directory %s added' %savedir)

SAVE_DATA = True

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02
Zsun   = 1.27E-02


datadir = '/orange/paul.torrey/SIMBA/groups/'

snaps = [
    151, 105, 79, 62, 51, 42, 36, 30, 26
]

auxtag = 'm100n1024'

for snap in snaps:
    print('Starting Snapshot %s' %snap)
    currentDir = savedir + 'snap%s/' %snap

    if not (path.exists(currentDir)):
        mkdir( currentDir )
        print('New directory %s added' %currentDir)

    fname = datadir + auxtag + "_" + str(snap).zfill(3) + '.hdf5'

    with h5py.File(fname, 'r') as f:    
        hdr = f['simulation_attributes']

        h   = hdr.attrs['hubble_constant']
        scf = hdr.attrs['time']
        z   = hdr.attrs['redshift']

        cat   = f['galaxy_data']
        dicts = cat['dicts']

        central = np.array(cat['central'])
        SFR     = np.array(cat['sfr'])
        Zgas    = np.array(dicts['metallicities.sfr_weighted'])
        Zstar   = np.array(dicts['metallicities.stellar'])
        Mstar   = np.array(dicts['masses.stellar']) ## Msun
        Mgas    = np.array(dicts['masses.gas']) ## Msun
        Rstar   = np.array(dicts['radii.stellar_half_mass'] * (scf)) ## kpc
        Rgas    = np.array(dicts['radii.gas_half_mass'] * (scf)) ## kpc

        keep_mask = ( (SFR > 0) & (Mstar > 1.00E+9) & (central) )

        print(sum(keep_mask))
        
        if SAVE_DATA:
            np.save( currentDir+'Zgas'        , np.array( Zgas  [keep_mask] ) )
            np.save( currentDir+'Zstar'       , np.array( Zstar [keep_mask] ) )
            np.save( currentDir+'SFR'         , np.array( SFR   [keep_mask] ) )
            np.save( currentDir+'Stellar_Mass', np.array( Mstar [keep_mask] ) )
            np.save( currentDir+'Gas_Mass'    , np.array( Mgas  [keep_mask] ) )
            np.save( currentDir+'R_gas'       , np.array( Rgas  [keep_mask] ) )
            np.save( currentDir+'R_star'      , np.array( Rstar [keep_mask] ) )
