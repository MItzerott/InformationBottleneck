import local_config
import argparse
#parse name of config file to be used here!!! I think that is the easiest way to deal with single runs in parallel in the cluster.
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--configname", type = str, help = "filename of the config.toml file with hyperparameters that should be used!")
args = parser.parse_args()

if args.configname:
    config_filename = args.configname
else:
    config_filename = 'config_blobs.toml'
    
local_config.config_filename = config_filename


from IB_refined import IB_refined
from data_generation_refined import gen_test_gaussians_labels
import time
from global_config import load_config
import os

def main(mode, smoothing_scale, output = False):
    import pandas as pd
    import numpy as np
    #ds = gen_test_gaussians_labels(plot = False, N = 10000, N_clusters = 20, box_size = 50000)
    ds = gen_test_gaussians_labels(
        N = 1000, 
        N_clusters = 4, 
        plot=True, 
        box_size = 50000
    )

    ds.s = np.float32(smoothing_scale)
    #ds.plot_coord(save = True)
    #multiple runs can be setup like this:
    #fit_param = pd.DataFrame(data={'alpha': [1, 1, 1, 1, 1, 1, 1, 1, 1], 'p0': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], 'waviness': [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]})
    
    #single run setup through config file.
    fit_param = pd.DataFrame(data={'alpha': [1]})
    fit_param['p0'] = load_config(filename = config_filename)['ib_fit_params']['p0']
    fit_param['waviness'] = load_config(filename = config_filename)['ib_fit_params']['waviness']
    fit_param['beta_search'] = bool(load_config(filename = config_filename)['ib_fit_params']['beta_search'])
    fit_param['max_fits'] = load_config(filename = config_filename)['ib_fit_params']['max_fits']
    fit_param['max_time'] = load_config(filename = config_filename)['ib_fit_params']['max_time']
    fit_param['repeats'] = load_config(filename = config_filename)['ib_fit_params']['repeats']
    fit_param['geoapprox'] = bool(load_config(filename = config_filename)['ib_fit_params']['geoapprox'])
    fit_param['clamp'] = bool(load_config(filename = config_filename)['ib_fit_params']['clamp'])
    fit_param['using_labels'] = bool(load_config(filename = config_filename)['ib_fit_params']['using_labels'])
    fit_param['Tmax'] = 1000
    # fit models

    if mode == 'own': metrics_conv, dist_conv, metrics_sw, dist_sw = IB_refined(ds, fit_param, conv_dist_to_keep = {'qt_x','qt'}, sw_dist_to_keep = False, quiet = True, output = output)
    #if mode == 'own': metrics_conv, dist_conv, metrics_sw, dist_sw = IB_own(ds, fit_param, conv_dist_to_keep = {'qt_x','qt', 'qy_t'}, sw_dist_to_keep = {'qt_x', 'qt', 'qy_t'}, quiet = True, output = output)

    return metrics_conv, dist_conv, metrics_sw, dist_sw

start_time = time.time()
metrics_conv_own, dist_conv_own, metrics_sw_own, dist_sw_own = main(mode = 'own', smoothing_scale = 1., output = True)
end_time = time.time()
print('Algorithm ran for %.2fs' %(end_time-start_time))
