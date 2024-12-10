import numpy as np
from scipy import optimize
from scipy import signal
from scipy import stats

def gaussian_irf(rep_rate, num_bins, fwhm):
    """Generate a gaussian irf with FWHM specified in ns"""
    
    t = np.linspace(0,1/(rep_rate * 10**6), num_bins)
    irf = stats.norm.pdf(t, loc=t[-1]/2, scale=10**-9*fwhm/2.355)
    irf *= 1.0/irf.max()

    return(irf, t)

def gen_signal(rep_rate, num_bins, contributions, taus, irf=None):
    """Generate a signal from a weighted sum of exponentials with given lifetime, and optionally convolve with an IRF.
    In accordance with eq. 14 from Liu et al. 2012
    
    Arguments:
        rep_rate: The laser repetition rate in MHz
        num_bins: the number of time bins in the final signal
        contributions: An ordered list of fractional contributions of each decay
        taus: An ordered list of lifetimes of each decay in ns
        irf: optional irf function for convolution with pure signal

    Outputs:
        sig: Synthetic noise-free signal normalized to a peak value of 1
        t: time values of each bin supporting sig in ns
        avg_lt: true average lifetime
    """

    t = np.linspace(0,1/(rep_rate * 10**6), num_bins)
    sig = np.zeros(np.size(t))

    for decay in zip(contributions, taus, strict=True):
        sig += decay[0] * np.exp(-10**9 * t/decay[1])
    
    if irf is not None:
        sig = signal.convolve(sig, irf, 'same')


    sig *= 1.0/sig.max()

    avg_lt = np.average(taus, weights=contributions)

    return sig, t, avg_lt

def add_white_noise(sig, noise_db):
    """Add white noise with a specified strength. The SNR in db is defined as 20 * log(mu/sigma), where mu is the average of the signal.
    """
    sd = np.mean(sig) * 1.0 / 10**(noise_db/20)
    
    return np.random.normal(sig, sd)