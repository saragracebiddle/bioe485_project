import numpy as np
from scipy import special

def single_basis(k, alpha, l):
    """Generate discrete-time laguerre polynomial according to the definition given in Liu et al. 2012
    
    args:
        k: integer sample number
        alpha: laguerre alpha defining rate constant
        l: laguerre order
    """

    prefactor = alpha**(0.5*(k-l)) * (1.0-alpha)**0.5
    postfactor = np.zeros(np.size(k))
    for i in range(l+1):
        postfactor += (-1)**i * special.binom(k,i)*special.binom(l,i) * alpha**(l-i) * (1-alpha)**i

    return prefactor * postfactor