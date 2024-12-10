from laguerre_basis import *

import numpy as np
from scipy.signal import convolve

def compute_V(N, alpha, L, iirf):
    """Compute V matrix 


    args:
        N: number of samples/time points
        alpha: laguerre alpha defining rate constant
        L: number of Laguerre basis polynomials to us
        iirf: instrument impulse response function in the form of
        a list of length N or array of dimensions (N,1).

    Returns:
        V:  np_array with shape (N, L) 
    """

    # k values from 0 to N
    k = np.arange(N)

    # compute basis array
    basis_array = np.vstack(
        [single_basis(k, alpha, l) for l in range(L)]
    )
    basis_array = np.transpose(basis_array)

    #eye = basis_array.dot(basis_array.T)
    #assert np.allclose(eye, np.identity(eye.shape[0]), rtol = 0.1, atol = 0.1)



    # ensure iirf array is correct shape
    iirf = np.reshape(iirf, (N,1))


    # check iirf and basis array have compatible shapes before multiplication
    #assert iirf.shape[0] == basis_array.shape[0]

    # caclulate V matrix
    V = convolve(basis_array, iirf, mode = 'same')
    #mult = np.reshape(iirf[::-1], (N,1)) * basis_array
    
    #V = np.cumsum(mult, axis = 0)
    # TODO replace lines 42 and 42 with scipy convolve(basis_arry, iirf, mode = 'same')
    # check V is the correct shape
    assert V.shape == (N, L)
    return V