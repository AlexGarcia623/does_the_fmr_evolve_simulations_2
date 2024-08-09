# EAGLE Subhalo Query
import numpy as np
import sys
from os import path, mkdir

## https://github.com/kyleaoman/eagleSqlTools
EAGLE_SQL_TOOLS = './eagleSqlTools'
sys.path.insert(1,EAGLE_SQL_TOOLS)
try:
    import eagleSqlTools as sql
except:
    raise Exception("You need to install EAGLE SQL tools: https://github.com/kyleaoman/eagleSqlTools")


#########################################################################
################# YOU WILL PROBABLY HAVE TO CHANGE THIS #################
savedir = './Data/EAGLE/'
#########################################################################

SAVE_DATA = False

if not (path.exists(savedir)) and SAVE_DATA:
    mkdir( savedir )
    print('New directory %s added' %savedir)

## Need to define connector
# con = sql.connect( ENTER_YOUR_USER_NAME, password=ENTER_YOUR_PASSWORD )
if !con:
    raise Exception('You need to make your own EAGLE username/password!: https://virgodb.dur.ac.uk/')

SIM = 'RefL0100n1504' ## Can change depending on what run you want
snaps = [28,19,15,12,10,8,6,5,4,3,2] ## Integer redshifts 0-10

for snap in snaps:  
    currentDir = savedir + 'snap%s/' %snap
    
    if not (path.exists(currentDir)) and SAVE_DATA:
        mkdir(currentDir)
        print('New directory %s added' %currentDir)
    
    ## See documentation in Appendecies of https://arxiv.org/pdf/1510.01320 
    myQuery = '''SELECT \
        SF_Metallicity as Zgas,\
        Stars_Metallicity as Zstar,\
        StarFormationRate as SFR,\
        MassType_Star as Stellar_Mass,\
        MassType_Gas as Gas_Mass,\
        HalfMassRad_Gas as R_gas,\
        HalfMassRad_Star as R_star\
    FROM \
        RefL0100N1504_SubHalo as SH\
    WHERE \
        SnapNum = %s \
        and SH.SubGroupNumber = 0 
        and SH.StarFormationRate > 0.0\
        and SH.MassType_Star > 1E8\
        and SH.SubGroupNumber = 0''' %(snap) 
    
    print('Starting Query... snapshot %s' %snap)
    myData = sql.execute_query(con, myQuery)
        
    if SAVE_DATA:
        print('\tSaving Data')
        np.save(currentDir+'Zgas'        , np.array(myData['Zgas'][:]) )
        np.save(currentDir+'Zstar'       , np.array(myData['Zstar'][:]) )
        np.save(currentDir+'SFR'         , np.array(myData['SFR'][:]) )
        np.save(currentDir+'Stellar_Mass', np.array(myData['Stellar_Mass'][:]) )
        np.save(currentDir+'Gas_Mass'    , np.array(myData['Gas_Mass'][:]))
        np.save(currentDir+'R_gas'       , np.array(myData['R_gas'][:]))
        np.save(currentDir+'R_star'      , np.array(myData['R_star'][:]))
    
    print('Query Complete')
    print('\n')
    
