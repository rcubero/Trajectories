from __future__ import division
import numpy as np
from scipy.signal import correlate2d

def phi(l):
    return -np.pi/6. + l*np.pi/3.
    
def firing_rate(x_, c_, n_max_, kappa_, lambda_0_, omega_, orientation_):
    k = np.array([(np.cos(phi(l + orientation_)),np.sin(phi(l + orientation_))) for l in [1,2,3]])
    return n_max_*np.exp((kappa_/3.)*np.sum(np.cos(omega_*np.dot(k,x_-c_))-1))

def autocorrelation_normalized(X):
    corr = correlate2d(X, X, boundary='fill', fillvalue=0.)
    one_corr = correlate2d(np.ones(np.shape(X)), np.ones(np.shape(X)), boundary='fill', fillvalue=0.)
    return np.divide(corr, one_corr)
