import numpy as np

def read_jla_data(sn_list_name = None):
    """
    read in table with the JLA supernova type Ia sample
    
    Parameters
    ----------
    sn_list_name: str
        path/file name containing the JLA data table in ASCII format
        
    Returns
    -------
    zCMB, mB, emB - numpy float vectors containing 
                       zCMB: SNIa redshifts in the CMB frame
                       mB, emB: apparent B-magnitude and its errors
    """
    zCMB, mB, emB = np.loadtxt(sn_list_name, usecols=(1, 4, 5),  unpack=True)

    return zCMB, mB, emB
