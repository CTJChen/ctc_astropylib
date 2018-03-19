import scipy.stats.distributions as dist
import numpy as np
from astropy.coordinates import Distance
from sklearn.neighbors import NearestNeighbors

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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def get_distnn(ra, dec, algorithm='auto'):
    '''
    Use sklearn.neighbors.NearestNeighbors
    to compute the distance to nearest neighbors for a set of RA and dec
    ra, dec should be in degrees (floats or doubles)
    the outputs are:
    distnn and idnn
    distnn is in arcsec by default.
    The default algorithm is auto, 
    but scikitlearn allows the following options:
    ['auto', 'ball_tree', 'kd_tree', 'brute']

    '''
    X = np.vstack((ra,dec)).transpose()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm=algorithm).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distnn = distances[:,1]*3600.
    idnn = indices[:,1]
    return distnn,idnn

