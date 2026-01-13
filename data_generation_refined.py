from IB_refined import *
import matplotlib.pyplot as plt
import os
import numpy as np
from model_refined import model_refined
from dataset_refined import dataset_refined
from read_snapshot import read_snapshot as rg
import numpy as np
from global_config import load_config

def mock_to_data(plot = True, all = False):
    import numpy as np
    from Read_Sim_Martin import ReadHaloFile
    npart, MassArray, Time, pos, vel, ID = ReadHaloFile()
    name = "mock_halo"
    
    print(np.max(pos), np.min(pos))
    # make labels
    labels = np.array(ID)
    # make coordinates
    
    coord = pos.reshape(pos.size//3, 3)
    boxsize = 50000
    
    print(pos.shape)
    print(coord.shape)

    print(ID.shape)
    print(npart, MassArray, Time)
    
    rng = np.random.default_rng()
    rng.shuffle(coord, axis = 0)
    #print(coord_shuffle)
    #labeld_shuffle = 
    # make dataset
    if all:
        ds = dataset_refined(coord = coord, labels = labels, name = name, box_size = boxsize)
    else:
        np.random.seed(42)
        rand_idx = np.random.randint(low = 0, high = coord.shape[0], size = 10000)
        ds = dataset_refined(coord = coord[rand_idx,:], labels = labels[rand_idx], name = name, box_size = boxsize)
        
    # plot coordinates
    if plot: ds.plot_coord_3D()

    # normalize
    #ds.normalize_coord()
    #if plot: ds.plot_coord()

    return ds

def snapshot_to_data(size, identifier, plot = False, all = True, smoothing_scale = 1, quiet = True):
    # set name
    import numpy as np
    name = "snapshot"

    pos, vel, pot, ids, h, z, boxsize, npart, mass_part = rg(sim_size = size, identifier = identifier, quiet=quiet)
    
    # make labels
    # perhaps scramble them a li'l
    shuffle = True
    if shuffle:
        rng = np.random.default_rng(42)  # modern, safer RNG
        idx = rng.permutation(len(ids))

        labels = np.array(ids[idx])       # no need for np.array() again
        coord  = np.array(pos[idx, :])    # keeps alignment with labels
    else:
        labels = np.array(ids)
        coord = np.array(pos)
        
    # make dataset
    if all:
        ds = dataset_refined(
            coord = coord, 
            labels = labels, 
            name = name, 
            box_size = boxsize,
            s = smoothing_scale, 
            total_bins = load_config(config_filename)['ib_model_params']['total_bins_1D']
        )
    else:
        ds = dataset_refined(
            coord = coord[:3000], 
            labels = labels[:3000], 
            name = name, 
            box_size = boxsize, 
            s = smoothing_scale, 
            total_bins = load_config(config_filename)['ib_model_params']['total_bins_1D']
        )
        
    # plot coordinates
    if plot: ds.plot_coord()

    # normalize
    #ds.normalize_coord()
    #if plot: ds.plot_coord()

    return ds

def gen_test_gaussians_labels(N, N_clusters, plot=True, box_size = 50000):
    np.random.seed(90)

    #N = 1000
    #N_clusters = 30
    box_size = 50000
    # set name
    name = "easytest"
    N = N - N%N_clusters #make sure, that the Datapoints are equally distriuted among clusters and none are left to be zero, since the number of datapoints is not dividable by the number of clusters.

    #number of datapoints per cluster
    n = N//N_clusters
    DIM = 3

    # set generative parameters  
    #random cluster means.
    mu = np.random.uniform(low=0, high=box_size, size=(N_clusters, DIM))
    #mu = np.array([[box_size, box_size, box_size], [box_size//2, box_size//2, box_size//2]])#, [box_size, box_size, 0], [0, box_size, box_size], [0, 0, 0]])

    #for now, set all sigmas to 1
    sig = np.zeros((DIM, DIM))
    for i in range(sig.shape[0]):
        for j in range(sig.shape[1]):
            #for k in range(sig.shape[2]):
            if(i==j): sig[i,j] = 1

    sig = np.identity(3)
    #and then scale the sigmas accordingly
    sig_scale = np.random.randint(low = 10000, high = 1000000, size = N_clusters)
    #sig_scale = np.ones(N_clusters) * 1000000
    param_2 = {}
    for i in range(N_clusters):
        param_2['mu%i' %(i+1)] = mu[i]
        param_2['sig%i' %(i+1)] = sig_scale[i] * sig
        param_2['n%i' %(i+1)] = n

    # make labels, such that afterwards, one can compare accuracy measures
    labels_2 = np.zeros(N, dtype = int)
    for k in range(N_clusters):
        labels_2[k*(n):(k+1)*n] = k

    Location = np.zeros((N, DIM))
    for i in range(N_clusters):
        Location[i*(n):(i+1)*(n),:] = np.random.multivariate_normal(mu[i], sig_scale[i] * sig, size = n)


    Location[Location > box_size] -= box_size#(Location[:,0] + np.max(Location[:,0]))%(np.max(Location[:,0]))
    Location[Location < 0] += box_size

    import h5py
    File = h5py.File('test_gaussians_rand_mu_unif_sigma_%i_%i.hdf5' %(N, N_clusters), 'w')
    File.create_dataset(name = 'Coordinates', data = Location)
    File.create_dataset(name = 'Labels', data = labels_2, dtype = np.int8)
    File.close()
    
    ds = dataset_refined(
        coord = Location, 
        labels = labels_2, 
        gen_param = None, 
        name = name, 
        box_size = box_size, 
        s = 1.0, 
        total_bins = load_config(config_filename)['ib_model_params']['total_bins_1D']
    )

    if plot: ds.plot_coord(save = True)
        
    return ds

def gen_BC_test(N, N_clusters = 1, plot=True, box_size = 50000):
    
    # set name
    name = "easytest"
    N = N - N%N_clusters #make sure, that the Datapoints are equally distriuted among clusters and none are left to be zero, since the number of datapoints is not dividable by the number of clusters.
    
    n = N//N_clusters
    DIM = 3
    # set generative parameters  
    #mu = np.zeros((N_clusters, DIM), dtype = np.int8)#np.random.uniform(low=-100, high=100, size=(N_clusters, DIM))
    mu = np.array([[box_size, box_size, box_size], [box_size//2, box_size//2, box_size//2]])#, [box_size, box_size, 0], [0, box_size, box_size], [0, 0, 0]])
    sig = np.zeros((DIM, DIM))
    for i in range(sig.shape[0]):
        for j in range(sig.shape[1]):
            #for k in range(sig.shape[2]):
            if(i==j): sig[i,j] = 1
    #print(sig)

    #sig_scale = np.random.randint(low = 1, high = 10, size = N_clusters)
    sig_scale = np.array([20, 10])#, 20, 20, 20])
    param_2 = {}
    for i in range(N_clusters):
        param_2['mu%i' %(i+1)] = mu[i]
        param_2['sig%i' %(i+1)] = sig_scale[i] * sig
        param_2['n%i' %(i+1)] = n

    #print(param)
    #print(param_2)
    # make labels
    labels_2 = np.zeros(N, dtype = int)
    for k in range(N_clusters):
        labels_2[k*(n):(k+1)*n] = k
    #labels[:n] = 0
    labels = np.array([0]*n+[1]*n+[2]*n)
    #print(labels, labels_2)
    # make coordinates
    #coord = np.concatenate((np.random.multivariate_normal(mu1,sig1,n1),
    #                        np.random.multivariate_normal(mu2,sig2,n2),
    #                        np.random.multivariate_normal(mu3,sig3,n3)))
    #sigma = np.array([.025, .025])
    covariance = np.diag(sig_scale**2)
    
    Location = np.zeros((N, DIM))
    for i in range(N_clusters):
        Location[i*(n):(i+1)*(n),:] = np.random.multivariate_normal(mu[i], sig_scale[i] * sig, size = n)
    #print(Location)
    #print(coord.shape, Location.shape)
    # make dataset_own

    
    ds = dataset_refined(coord = Location, labels = labels_2, gen_param = None, name = name, box_size = box_size)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    
    Location[Location > box_size] -= box_size#(Location[:,0] + np.max(Location[:,0]))%(np.max(Location[:,0]))
    Location[Location < 0] += box_size
    ds = dataset_refined(coord = Location, labels = labels_2, gen_param = None, name = name, box_size = box_size)

    if plot: ds.plot_coord()
    print(box_size)
    # normalize
    #ds.normalize_coord()
    #f plot: ds.plot_coord()
        
    return ds
