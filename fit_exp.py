import numpy as np
from scipy import optimize

def single_exp(t,a1,k1):
        return a1 * np.exp(-10**9 * k1*t)
    
def double_exp(t, a1, k1, a2, k2):
    return a1 * np.exp(-10**9 * k1*t) + a2 * np.exp(-10**9 * k2*t)
    
def triple_exp(t, a1, k1, a2, k2, a3, k3):
    return a1 * np.exp(-10**9 * k1*t) + a2 * np.exp(-10**9 * k2*t) + a3 * np.exp(-10**9 * k3*t)

def fit_exp(t, y, numTerms, set_lifetimes = None):
    assert numTerms <= 3 & numTerms > 0


    #truncate to only fit past max point
    maxIdx = np.argmax(y)
    t = t[maxIdx:]
    y = y[maxIdx:]

    p_single = (1,0.4)
    p_double = (0.5,0.2,0.5,2.0)
    p_triple = (0.45,0.2,0.45,0.5,0.1,2.0)
    
    tout = []

    if set_lifetimes is not None:
        #convert to rate constant
        set_lifetimes = 1.0/np.array(set_lifetimes)
        match numTerms:
            case 1:
                popt, pcov = optimize.curve_fit(lambda t, a1: single_exp(t, a1, set_lifetimes[0]), t, y, p0=p_single[0])
                a = popt[0]
            case 2:
                popt, pcov = optimize.curve_fit(lambda t, a1 , a2 : double_exp(t, a1, set_lifetimes[0], a2, set_lifetimes[1]), t, y, p0=(0.5, 0.5), bounds=(0, [np.inf,np.inf]))
                a = (popt[0], popt[1])
            case 3:
                popt, pcov = optimize.curve_fit(lambda t, a1 , a2, a3 : triple_exp(t, a1, set_lifetimes[0], a2, set_lifetimes[1], a3, set_lifetimes[2]), t, y, p0=(0.5, 0.5, 0.5), bounds=(0, [np.inf, np.inf]))
                a = (popt[0], popt[1], popt[2])
    else:
        match numTerms:
            case 1:
                popt, pcov = optimize.curve_fit(single_exp, t, y, p0=p_single, bounds=(0, [np.inf, np.inf]))
                a = popt[0]
                tout = 1/popt[1]
            case 2:
                popt, pcov = optimize.curve_fit(double_exp, t, y, p0=p_double, bounds=(0, [np.inf,np.inf,np.inf,np.inf]))
                a = (popt[0], popt[2])
                tout = (1/popt[1], 1/popt[3])
            case 3:
                popt, pcov = optimize.curve_fit(triple_exp, t, y, p0=p_triple, bounds=(0, [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
                a = (popt[0], popt[2], popt[4])
                tout = (1/popt[1], 1/popt[3], 1/popt[5])
        
    return a, tout, np.sqrt(np.diag(pcov))