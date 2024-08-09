import numpy as np
import sys
import h5py
import illustris_python as il
from os import path, mkdir

WHICH_SIM = "tng".upper()
SF_WEIGHT = True

SAVE_DATA = False

#########################################################################
################# YOU WILL PROBABLY HAVE TO CHANGE THIS #################
savedir = '../blue_FMR/%s/data/' %WHICH_SIM
#########################################################################

if not (path.exists(savedir)):
    mkdir( savedir )
    print('New directory %s added' %savedir)

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02
Zsun   = 1.27E-02

print
print("#"*100)
print('Getting data for %s' %WHICH_SIM)
print("#"*100)
print

if (WHICH_SIM == "ORIGINAL" or WHICH_SIM == "ILLUSTRIS"):
    run       = 'L75n1820FP'
    base      = '/orange/paul.torrey/Illustris/Runs/' + run + '/'
    out_dir   = base
    snaps = [135,86,68,60,54,49,45,41,38,35,32]
    
    if (WHICH_SIM=="ILLUSTRIS"):
        WHICH_SIM = "ORIGINAL"
        
elif (WHICH_SIM == "ORIGINAL-2"):
    run       = 'L75n910FP'
    base      = '/orange/paul.torrey/alexgarcia/' + run + '/'
    out_dir   = base
    snaps = [135,86,68,60,54,49,45,41,38]
    
elif (WHICH_SIM == "ORIGINAL-3"):
    run       = 'L75n455FP'
    base      = '/orange/paul.torrey/alexgarcia/' + run + '/'
    out_dir   = base
    snaps = [135,86,68,60,54,49,45,41,38]
    
elif (WHICH_SIM == "TNG50"):
    run     = 'L35n2160TNG'
    ### Directories for snapshot 99 and 50 are different... for some reason
    # base    = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/'
    # out_dir = base
    # snaps   = [99]
    ###
    base    = '/orange/paul.torrey/zhemler/IllustrisTNG/' + run + '/'
    out_dir = base + 'output/'
    snaps   = [50]
    
elif (WHICH_SIM == "TNG50-2"):
    run     = 'L35n1080TNG'
    base    = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/'
    out_dir = base
    snaps   = [99,50]
    
elif (WHICH_SIM == "TNG"):
    run       = 'L75n1820TNG'
    base      = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/' 
    out_dir   = base
    snaps = [99,50,33,25,21,17,13,11,8,6,4]
    
elif (WHICH_SIM == "TNG-2"):
    run       = 'L75n910TNG'
    base      = '/orange/paul.torrey/alexgarcia/' + run + '/'
    out_dir   = base
    snaps = [99,50,33,25,21,17,13,11,8]
    
elif (WHICH_SIM == "TNG-3"):
    run       = 'L75n455TNG'
    base      = '/orange/paul.torrey/alexgarcia/' + run + '/'
    out_dir   = base
    snaps = [99,50,33,25,21,17,13,11,8]
    
m_star_min = 8.0
m_star_max = 12.0
m_gas_min  = 8.5

for snap in snaps:
    print('Starting snap %s' %snap)
    currentDir = savedir + 'snap%s/' %snap

    if not (path.exists(currentDir)):
        mkdir( currentDir )
        print('New directory %s added' %currentDir)
        
    print('Header')
    hdr  = il.groupcat.loadHeader(out_dir, snap)
    box_size = hdr['BoxSize']
    scf      = hdr['Time']
    z        = (1.00E+00 / scf - 1.00E+00)
    print(scf, z)

    print('Subhalos')
    fields = ['SubhaloMassType','SubhaloSFR','SubhaloStarMetallicity','SubhaloGasMetallicity','SubhaloHalfmassRadType','SubhaloGasMetallicitySfr']
    sub_cat = il.groupcat.loadSubhalos(out_dir,snap,fields=fields)

    print('Halos')
    fields  = ['GroupFirstSub','Group_R_Crit200']
    grp_cat = il.groupcat.loadHalos(out_dir,snap,fields=fields)

    subs = grp_cat['GroupFirstSub']

    subs = subs[(subs != 4294967295)] # Illustris weirdness...?

    # print (sub_cat['SubhaloMassType']).shape
    gas_mass  = sub_cat['SubhaloMassType'][subs,0] * 1.00E+10 / h
    star_mass = sub_cat['SubhaloMassType'][subs,4] * 1.00E+10 / h
    SFR       = sub_cat['SubhaloSFR'][subs]
    Zstar     = sub_cat['SubhaloStarMetallicity'][subs]
    if (SF_WEIGHT):
        Zgas  = sub_cat['SubhaloGasMetallicitySfr'][subs] # Star forming gas only
    else:
        Zgas  = sub_cat['SubhaloGasMetallicity' ][subs] 
    R_gas     = sub_cat['SubhaloHalfmassRadType'][subs,0] * (scf / h)
    R_star    = sub_cat['SubhaloHalfmassRadType'][subs,4] * (scf / h)

    keep_mask = ( (SFR > 0) & (star_mass > 1.00E+8) )
        
    if SAVE_DATA:
        np.save( currentDir+'Zgas'        , np.array( Zgas      [keep_mask] ) )
        np.save( currentDir+'Zstar'       , np.array( Zstar     [keep_mask] ) )
        np.save( currentDir+'SFR'         , np.array( SFR       [keep_mask] ) )
        np.save( currentDir+'Stellar_Mass', np.array( star_mass [keep_mask] ) )
        np.save( currentDir+'Gas_Mass'    , np.array( gas_mass  [keep_mask] ) )
        np.save( currentDir+'R_gas'       , np.array( R_gas     [keep_mask] ) )
        np.save( currentDir+'R_star'      , np.array( R_star    [keep_mask] ) )


    print('Done with snap %s' %snap)
    print('\n')
