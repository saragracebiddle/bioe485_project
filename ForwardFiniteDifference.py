import numpy as np
from laguerre_basis import * 
from scipy.sparse import diags

def Build_D(k_samples):
    """Build the third order forward finite difference matrix
    
    args:
        K_samples: number of samples/time points
   
    """
    
    stencil = [-1, 3, -3, 1]
    # Create diagonals for the sparse matrix
    n=k_samples
    stencil = [-1, 3, -3, 1]  # Stencil coefficients
    diagonals = [
        np.full(n - 3, stencil[0]),  # Main diagonal
        np.full(n - 3, stencil[1]),  # First upper diagonal
        np.full(n - 3, stencil[2]),  # Second upper diagonal
        np.full(n - 3, stencil[3])   # Third upper diagonal
    ]
    #print(diagonals)
    # Construct the sparse matrix
    D3 = diags(diagonals, [0,1,2,3], shape=(n-3,n)).toarray()
    return D3

def Compute_ForwardFiniteDiff(numBasis, samples, a):
    """Compute the third order forward finite difference of laguerre basis functions
    args:
        numBasis: number of laguerre basis to compute (laguerre order)
        samples: number of samples/time points
        alpha: laguerre alpha defining rate constant
    """
    k=np.arange(samples)
    basis = single_basis(k, a, numBasis) #get the Lth order basis
    #print(basis.T.shape)
    D = Build_D(samples)                 #build the third order finite difference matrix
    FFD = D@basis                        #Multiply the matrices
    return FFD
        