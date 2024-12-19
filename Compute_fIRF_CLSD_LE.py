import numpy as np
from laguerre_basis import *
from compute_V import *
from cholesky_decomp import *
from ForwardFiniteDifference import *
from scipy.optimize import minimize


def objectiveFunc(lmbda, C, V, sig_irf, FFD3):
    """Define the objective function that will be used by our minimizer
    
    args:
        lmbda: Free parameter we wish to optimize
        C: Cholesky Decomposition result
        V: computed V as described in liu 2012
        sig_irf: generated signal data 
        FFD3: third order finite difference result
   
    """
    lmbda = np.array(lmbda)  # Ensure lambda is an array
    residual = C @ (V.T @ sig_irf - FFD3.T @ lmbda)
    return np.linalg.norm(residual)**2

def CLSD_minimize(lmbda, C, V, sig_irf, FFD3):
    """Call the minimization function
    
    args:
        C: Cholesky Decomposition result
        V: computed V as described by equaton 6 from Liu et al. 2012
        sig_irf: generated signal data 
        FFD3: third order finite difference result
    """
    lmbda = np.array(lmbda)  # Ensure lambda is an array
    lmbda_optimal = minimize(objectiveFunc,x0=lmbda,args=(C, V, sig_irf, FFD3),method='trust-constr')
    return lmbda_optimal


def compute_Ccls(V,C,FFD3,lmbda_optimal,sig_irf):
    """compute laguerre coefficents as desbribed by equation 12 from Liu et al. 2012
    
    args:
        V: computed V as described in liu 2012
        C: Cholesky Decomposition result
        FFD3: third order finite difference result
        lmbda: optimized lagragian parameter
    """
    coefficients = np.linalg.pinv(V.T@V)@(V.T@sig_irf-FFD3.T@np.abs(lmbda_optimal.x))
    return coefficients

#Find impulse response
def ComputefIRF(coeffs, samples, a, numBasis):
    """compute the fIRF as described by equaton 12 from Liu et al. 2012
    
    args:
        coeffs: calculated coefficients with optimized lambda parameter
        sample: number of samples/time points
        FFD3: third order finite difference result
        a: laguerre alpha defining rate constant
        numBasis: number of laguerre basis to compute (laguerre order)
    """
    hfIRF=0
    k = np.arange(samples)
    for l in range(numBasis):
        hfIRF += single_basis(k,a,numBasis)*coeffs[l]
        #print(hfIRF)

    hfIRF *= 1.0/hfIRF.max() #normalize

    return hfIRF




