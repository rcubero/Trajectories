# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import os, sys

# some additional packages that are needed
from scipy import io, signal
from multiprocessing import Pool
import itertools


# import external dictionaries
from preprocess import *
from relevance import *
from spatial_quantities import *
from misc import *
import gridcell
import randomwalk


rand_seed = sys.argv[1]

# Load the positions and calculate speeds
pos = io.loadmat('BEN_pos.mat', squeeze_me=True)
positions = np.array([pos['post'],(pos['posx']+pos['posx2'])/2.,(pos['posy']+pos['posy2'])/2.])

tight_range = ((-74.5, 74.5), (-74.5, 74.5))
positions[1], positions[2], info = transform(positions[1],positions[2],range_=tight_range,translate=True,rotate=True)

binning_time = 0.01
time_bins = np.arange(positions[0][0], positions[0][-1]+binning_time, binning_time)
x_t = np.interp(time_bins, positions[0], positions[1]); y_t = np.interp(time_bins, positions[0], positions[2]) # Interpolate the midpoint positions of the LEDs


kappa_range = [0.1, 0.25, 0.5, 0.75, 0.85, 1.0, 1.5, 2.0, 2.5, 3.0]
orientation_range = [l*np.pi/12. for l in np.arange(12)]
parameters = list(itertools.product(kappa_range, orientation_range))

n_max = 10 # peak firing rate
lambda_0 = sys.argv[2] # grid spacing
omega = (2.*np.pi)/(np.sin(np.pi/3.)*float(lambda_0))
c = np.array([30,0]) # grid phase


for rng_seed in np.arange(int(rand_seed), int(rand_seed)+50, 1):
    np.random.seed(int(rng_seed))
    results = {}

    results['parameters'] = parameters
    results['x_t'] = x_t
    results['y_t'] = y_t

    relevance = np.zeros(len(parameters))
    spatial_info = np.zeros(len(parameters))
    logM = np.zeros(len(parameters))
    for index in np.arange(len(parameters)):
        kappa, orientation = parameters[index]

        spike_data = np.array([ np.random.poisson(gridcell.firing_rate(np.array([x_t[i],y_t[i]]), c, 20, kappa, float(lambda_0), omega, orientation)*binning_time, 1)[0] for i in np.arange(len(x_t))])

        relevance[index] = parallelized_total_relevance((len(spike_data), spike_data))
        spatial_info[index] = spatial_information(1, spike_data,  x_t, y_t, binning_time)
        logM[index] = np.log(np.sum(spike_data))


    results['relevance'] = relevance
    results['HD_info'] = spatial_info
    results['logM'] = logM
    
    io.savemat('GridCell_Module%d_realtrajectory_seed%05d.mat'%(int(lambda_0), int(rng_seed)), results)

