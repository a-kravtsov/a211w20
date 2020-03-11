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

def Rmm(a, b, func, m, **kwargs):
    """
    Auxiliary function computing tableau entries for Romberg integration
    using recursive relation, but implemented non-recursively
    
    Parameters
    -----------------
    func - python function object
            function to integrate
    a, b - floats
            integration interval            
    m    - integer
            iteration level; accuracy order will be equal to 2(m+1)
            in this implementation there is no need for k on input
            
    kwargs - python dictionary 
            array of keyword arguments to be passed to the integrated function
            
    Returns
    ---------
    
    I(m)   - float
              estimate of the integral using scheme of order 2*m+2
    I(m-1) - float
              estimate of the integral using scheme of order 2*m
    """
    assert(m >= 0)
    
    ba = b - a;
    hk = ba / 2**(np.arange(m+1)) # vector of step sizes

    Rkm = np.zeros((m+1,m+1)) 

    Rkm[0,0] = 0.5 * ba * (func(a, **kwargs) + func(b, **kwargs))
        
    for k in range(1,m+1):
        # first compute R[k,0]
        trapzd_sum = 0.
        for i in range(1, 2**(k-1)+1):
            trapzd_sum += func(a + (2*i-1)*hk[k], **kwargs)
            
        # we can reuse Rkm[k-1,0] but we need to divide it by 2 to account for step decrease 
        Rkm[k,0] = Rkm[k-1,0] * 0.5 + hk[k] * trapzd_sum
        
        # then fill the tableau up to R[k,k]
        for md in range(1,k+1):
            fact = 4.**md
            Rkm[k,md] = (fact * Rkm[k,md-1] - Rkm[k-1,md-1])/(fact - 1)

          
    return Rkm[m,m], Rkm[m,m-1] # return the desired approximation and best one of previous order 

def romberg(func, a, b, rtol = 1.e-4, mmax = 8, verbose = False, **kwargs):
    """
    Romberg integration scheme to evaluate
            int_a^b func(x)dx 
    using recursive relation to produce higher and higher order approximations
    
    Code iterates from m=0, increasing m by 1 on each iteration.
    Each iteration computes the integral using scheme of 2(m+2) order of accuracy 
    Routine checks the difference between approximations of successive orders
    to estimate error and stops when a desired relative accuracy 
    tolerance is reached.
    
    - Andrey Kravtsov, 2017

    Parameters
    --------------------------------
    
    func - python function object
            function to integrate
    a, b - floats
            integration interval
    rtol - float 
            fractional tolerance of the integral estimate
    mmax - integer
            maximum number of iterations to do 
    verbose - logical
            if True print intermediate info for each iteration
    kwargs - python dictionary
             a list of parameters with their keywords to pass to func
               
    Returns
    ---------------------------------
    I    - float
           estimate of the integral for input f, [a,b] and rtol
    err  - float 
           estimated fractional error of the estimated integral

    """
    assert(a < b)
    
    for m in range(1, mmax):
        Rmk_m, Rmk_m1 = Rmm(a, b, func, m, **kwargs)
            
        if Rmk_m == 0:
            Rmk_m = 1.e-300 # guard against division by 0 
            
        etol = 1.2e-16 + rtol*np.abs(Rmk_m)
        err = np.abs(Rmk_m-Rmk_m1)

        if verbose: 
            print("m = %d, integral = %.6e, prev. order = %.6e, frac. error estimate = %.3e"%(m, Rmk_m, Rmk_m1, err/Rmk_m))

        if (m>0) and (np.abs(err) <= etol):
            return Rmk_m, err/Rmk_m
        
    print("!!! Romberg warning: !!!")
    print("!!! maximum of mmax=%d iterations reached, abs(err)=%.3e, > required error rtol = %.3e"%(mmax, np.abs(err/Rmk_m), rtol))
    return Rmk_m, err/Rmk_m
    

def d_func(a, **kwargs):
    """
    auxiliary function for the integrand of the comoving distance
    
    parameters:
    -----------
    a: float
       expansion factor of the epoch to which to compute comoving distance
    kwargs: keyword dictionary
        containing values of Om0, OmL, and Omk
    """
    a2i = 1./(a * a) 
    a2Hai = a2i / np.sqrt(kwargs["Om0"]/a**3 + kwargs["OmL"] + kwargs["Omk"] * a2i)
    return a2Hai
    
from scipy.interpolate import UnivariateSpline
    
def dcom(z, Om0, OmL, ninter=20):
    """
    function computing comoving distance Dc for a given redshift and mean matter and vacuum energies Om0 and OmL
    """
    Omk = 1. - Om0 - OmL
    kwargs = {"Om0": Om0, "OmL": OmL, "Omk": Omk}
    a = 1. / (1.0 + z)
    
    nz = np.size(z)
    if nz == 1:
        if np.abs(a-1.0) < 1.e-10:
            dc = 0.
        else:
            dc = romberg(d_func, a, 1., rtol = 1.e-10, mmax = 16, verbose = False, **kwargs)[0]
    elif nz > 1:
        dc = np.zeros(nz)
        if nz <= ninter:
            for i, ad in enumerate(a):
                if np.abs(ad-1.0) < 1.e-10:
                    dc[i] = 0.
                else:
                    dc[i] = romberg(d_func, ad, 1., rtol = 1.e-10, mmax = 16, verbose = False, **kwargs)[0]
        else:
            zmin = np.min(z); zmax = np.max(z)
            zi = np.linspace(zmin, zmax, num=ninter)
            fi = np.zeros(ninter)
            for i, zid in enumerate(zi):
                aid = 1.0/(1+zid)
                if np.abs(aid-1.0) < 1.e-10:
                    fi[i] = 0.
                else:
                    fi[i] = romberg(d_func, aid, 1., rtol = 1.e-10, mmax = 16, verbose = False, **kwargs)[0]
            dsp = UnivariateSpline(zi, fi, s=0.)
            dc = dsp(z)
    return dc
    
def d_l(z, Om0, OmL, ninter=20):
    """
    function computing luminosity distance
    
    parameters:
    -----------
    z: float
        redshift
    Om0: mean density of matter in units of critical density at z=0
    OmL: density of vacuum energy in units of critical density
    nspline: integer
        number of spline nodes in redshift to use for interpolation if number of computed distances is > nsp
        
    returns:
    --------
    d_L in units of d_H=c/H0 (i.e. to get distance in Mpc multiply by 2997.92
    """
    
    Omk = 1. - Om0 - OmL
    zp1 = 1.0 + z
    dc = dcom(z, Om0, OmL, ninter)    
    if np.abs(Omk) < 1.e-15:
        return dc * zp1
    elif Omk > 0:
        sqrtOmk = np.sqrt(Omk)
        return np.sinh(sqrtOmk * dc) / sqrtOmk * zp1
    else:
        sqrtOmk = np.sqrt(-Omk)
        return np.sin(sqrtOmk * dc) / sqrtOmk * zp1
        
def d_a(z, Om0, OmL):
    zp1i = 1./(z + 1.)
    return d_l(z, Om0, OmL) * zp1i * zp1i

