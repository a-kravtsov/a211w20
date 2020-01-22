import numpy as np
from .constants import clight 

# function that works only for models with OmL = 0
def d_L_simple(z, H0, Om0):
    q0 = 0.5 * Om0
    q0z = q0 * z
    return clight * z/H0 * (1. + (z-q0z) / (1. + q0z + np.sqrt(1. + 2.*q0z)))  

def dl_func(z, H0, Om0, OmL, Omk):
    z1 = 1.0 + z; z12 = z1 * z1
    return 1.0 / np.sqrt(z12*(Om0*z1 + Omk) + OmL)
    
from scipy.integrate import romberg

def d_L_romberg(z, H0, Om0, OmL, atol=1.e-8, rtol=1.e-8):
    if 1.0+OmL == 1.0:
        return d_L_simple(z, H0, Om0)
    else:
        dH = clight / H0 
        Omk = 1.0 - Om0 - OmL
        dc = romberg(dl_func, 0., z, args=(H0, Om0, OmL, Omk), divmax=15, tol=atol, rtol=rtol)
        if 1.0 + Omk == 1.0:
            return dH * dc * (1.0 + z)
        else:
            sqrOmk = np.sqrt(np.abs(Omk))
            if Omk > 0.:
                return dH * np.sinh(dc*sqrOmk) * (1.0 + z) / sqrOmk 
            else:
                return dH * np.sin(dc*sqrOmk) * (1.0 + z) / sqrOmk
            
        return
        
def vectorize_func(func, *x):
    """ Helper function to vectorize function with array inputs"""
    return np.vectorize(func)(*x)
    
def _dc_vec(z1, z2, *args, atol=1.e-8, rtol = 1.e-8):
    f = lambda z1, z2: romberg(dl_func, z1, z2, args=args, tol=atol, rtol=rtol)
    if np.size(z1) > 1 or np.size(z2) > 1:
        return vectorize_func(f, z1, z2)
    else:
        return f(z1, z2)
        
def d_L_vectorized(z, H0, Om0, OmL, atol=1.e-8, rtol=1.e-8):
    if 1.0+OmL == 1.0:
        return d_L_simple(z, H0, Om0)
    else:
        dH = clight / H0 
        Omk = 1.0 - Om0 - OmL
        args = [H0, Om0, OmL, Omk]
        dc = _dc_vec(0., z, *args, atol=atol, rtol=rtol)
        if 1.0 + Omk == 1.0:
            return dH * dc * (1.0 + z)
        else:
            sqrOmk = np.sqrt(np.abs(Omk))
            if Omk > 0.:
                return dH * np.sinh(dc*sqrOmk) * (1.0 + z) / sqrOmk 
            else:
                return dH * np.sin(dc*sqrOmk) * (1.0 + z) / sqrOmk
            
        return dL
