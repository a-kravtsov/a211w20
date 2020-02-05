import numpy as np

# functions generating N+1 Chebyshev nodes of the 1st and 2nd kind 
# for input N, and interval edges a and b

def chebyshev_nodes1(a, b, N):
    assert(b>a)
    return a + 0.5*(b-a)*(1. + np.cos((2.*np.arange(N+1)+1)*np.pi/(2.*(N+1))))

def chebyshev_nodes2(a, b, N):
    assert(b>a)
    return a + 0.5*(b-a)*(1. + np.cos(np.arange(N+1)*np.pi/N))
    
