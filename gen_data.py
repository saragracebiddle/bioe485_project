from sim_data import *
import numpy as np

# set seed for reproducibility
np.random.seed(23750)

# iIRF parameters
rep_rate = 40
k = 800
fwhm = 0.2

# iIRF only needs to be generated once
# it is measured one time and the same for all signals
iirf, t = gaussian_irf(rep_rate, k, fwhm)

# number of sets to generate
N = 1000

# fractional contributions and taus randomly generated
# in this case, based on the information in the paper,
# fractional contributions are generated from a uniform distribution between 
# 0 and 100%
# and taus are randomly generated from a uniform distribution between 1 and 6 ns
# and the number of exponential components generated is randomly picked from 1-6
rng = np.random.default_rng(12345)
M = rng.integers(low=1, high=6, size=N)
# TODO order fracs and taus correctly for input into gen_signal
# seems to work without it, double check later
parts = [rng.random(m) for m in M]
fracs = [parts[n] / sum(parts[n]) for n in range(N)]
taus = [rng.uniform(low = 1, high = 6, size = m) for m in M]

# generate 1000 signals
decays = []
for n in range(N):
    decay, t = gen_signal(rep_rate, k, fracs[n], taus[n], iirf)
    noisy_decay = add_white_noise(decay, 20)
    decays.append(noisy_decay)

# split into testing, training, and validation
# data is already randomly generated, don't really need to shuffle unless we want to
train_data = decays[:800]
test_data = decays[800:]