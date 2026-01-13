import numpy as np
from kl_divergence import kl_div_parallel_sparse
from entropy import entropy_parallel
import time
from global_config import load_config
import local_config

config_filename = local_config.config_filename
print_func_names = False

def cutoff(a, limit = 1e-12, upper_limit = 1):
    a = np.clip(a, limit, upper_limit)
    return a

def entropy(P, limit = 1e-12):
    """Returns entropy of a distribution, or series of distributions.
    
    For the input array P [=] M x N, treats each col as a prob distribution
    (over M elements), and thus returns N entropies. If P is a vector, treats
    P as a single distribution and returns its entropy."""
    import time
    startentr = time.time()
    P = P.astype(np.float32)
    logP = np.zeros(P.shape, dtype = np.float32)
    logP[P > limit] = np.log2(P[P > limit])
    logP = logP.astype(np.float32)
        
    if print_func_names: print('entropy_own')
    P = cutoff(P)
    if P.ndim==1: return np.nansum(np.where((P > 0), -P * np.log2(P), 0.0))
    else:
        H = entropy_parallel(P, logP, num_threads = load_config(filename = config_filename)['ib_code_params']['max_n_cpus'])
        return H

def kullback_leibler(P, Q, MAX_PROCESSES = load_config(filename = config_filename)['ib_code_params']['max_n_cpus'], benchmark = False, limit = 1e-12):
    """Returns KL divergence of one or more pairs of distributions.
    
    For the input arrays P [=] M x N and Q [=] M x L, calculates KL of each col
    of P with each col of Q, yielding the KL matrix DKL [=] N x L. If P=Q=1,
    returns a single KL divergence."""
    if print_func_names: print('kl_own')
    if P.ndim==1 and Q.ndim==1: return np.sum(np.where((P > 0, Q > 0), P * np.log2(P/Q), 0.0))
    elif P.ndim==1 and Q.ndim!=1: # handle vector P case
        M = len(P)
        N = 1
        M2,L = Q.shape
        if M!=M2: raise ValueError("P and Q must have same number of columns")
        DKL_2 = np.nansum(np.where((P > 0, Q > 0), P * np.log2(P/Q, axis = 0), 0.0), axis = 0)
        
    elif P.ndim!=1 and Q.ndim==1: # handle vector Q case
        M,N = P.shape
        M2 = len(Q)
        L = 1
        if M!=M2: raise ValueError("P and Q must have same number of rows")
        DKL_2 = np.nansum(np.where((P > 0, Q > 0), P * np.log2(P/Q, axis = 1), 0.0), axis = 1)

    else:
        if P.shape[0] != Q.shape[0]:
            raise ValueError("P and Q must have the same number of rows")

        DKL_cython = np.zeros((Q.shape[1], P.shape[1]))
        import os
        
        N_proc = min((MAX_PROCESSES, os.cpu_count(), P.shape[1]))
        print(P.shape, Q.shape, N_proc)
        
        nonzero_rows = np.full((P.shape[1], P.shape[0]), dtype = np.int32, fill_value= P.shape[0]+1)
        size_rows = np.zeros(P.shape[1], dtype = np.int32)

        for i in range(P.shape[1]):
            idx = np.argwhere(P[:,i] > limit).flatten().astype(np.int32)
            nonzero_rows[i,:len(idx)] = idx
            size_rows[i] = len(idx)

        nonzero_rows = np.array(nonzero_rows)

        DataType = np.float32
        logP = np.zeros(P.shape, dtype = DataType)
        logQ = np.zeros(Q.shape, dtype = DataType)
        logP[P > limit] = np.log2(P[P > limit])
        logQ[Q > limit] = np.log2(Q[Q > limit])
        logP = logP.astype(DataType)
        logQ = logQ.astype(DataType)
        P.astype(DataType)
        Q.astype(DataType)

        if benchmark:
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = 1)[0]
            round1 = time.time() - start
            print('Run complete')
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = 2)[0]
            round2 = time.time() - start
            print('Run complete')
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = 4)[0]
            round4 = time.time() - start
            print('Run complete')
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = 8)[0]
            round8 = time.time() - start
            print('Run complete')
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = 16)[0]
            round16 = time.time() - start
            print('Run complete')
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = 32)[0]
            round32 = time.time() - start
            print('Run complete')
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = 64)[0]
            round64 = time.time() - start
            print('Run complete')
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = 128)[0]
            round128 = time.time() - start
            print('Runtime: 1 Thread: %.2f, 2 Threads: %.2f, 4 Threads: %.2fs, 8 Threads: %.2fs, 16 Threads: %.2fs, 32 Threads: %.2fs, 64 Threads: %.2fs, 128 Threads: %.2fs' 
                %(round1, round2, round4, round8, round16, round32, round64, round128))
        
        else:
            start = time.time()
            DKL_cython[:,:] = kl_div_parallel_sparse(P.astype(DataType), Q.astype(DataType), logP, logQ, nonzero_rows, size_rows, num_threads = N_proc)[0]
            round32 = time.time() - start
            print('Runtime: %.2fs' %round32, flush = True)

    return DKL_cython - np.min(DKL_cython) #-min gets rid of numerical uncertainty, where KL can be <0 in some cases.
