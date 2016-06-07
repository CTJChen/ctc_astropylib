import scipy.stats.distributions as dist
import numpy as np
from astropy.coordinates import Distance 

def bayes_ci(k, n, sigma=None):
    '''
    Calculate confidence interval using the binomial
    distribution/bayesian methods described in Cameron et al. 2011
    '''
    sig = {'1': 0.68268949, '2': 0.95449974,
           '3': 0.99730020, '4': 0.99993666, '5': 0.99999943}
    if sigma is None:
        c = 0.683
    elif sigma in sig:
        c = sig[sigma]
    else:
        return 'sigma = 1~5 only'
    err_lower = k/n - dist.beta.ppf((1-c)/2., k+1, n-k+1)
    err_upper = dist.beta.ppf(1-(1-c)/2., k+1, n-k+1) - k/n
    return np.array([err_lower, err_upper])


def dmod(redshift,distance=None):
    if distance is not None:
        dist = distance.to(parsec).value/10.
    else:
        dist = Distance(z=redshift).parsec/10.
    dm=5*np.log10(dist-5)
    return dm