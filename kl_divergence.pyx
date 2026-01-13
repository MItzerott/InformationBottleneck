# distutils: language=c++

from cython.cimports.libc import math
from cython.parallel import prange
from libc.math cimport log2
import numpy as np
cimport numpy as cnp
#from scipy import sparse

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <sched.h>
#include <omp.h>

from cython.parallel import parallel, prange
cimport cython
#include <iostream> 

@cython.boundscheck(False)
@cython.wraparound(False)
    
def kl_div_parallel_sparse(cnp.ndarray[cnp.float32_t, ndim=2] P, cnp.ndarray[cnp.float32_t, ndim=2] Q, cnp.ndarray[cnp.float32_t, ndim=2] log2P, cnp.ndarray[cnp.float32_t, ndim=2] log2Q, cnp.ndarray[int, ndim = 2] non_zero_bins, cnp.ndarray[int, ndim = 1] sizes, int num_threads):
    cdef int bins = P.shape[0]
    cdef int particles = P.shape[1]
    cdef int clusters = Q.shape[1]
    cdef int i, j, k
    cdef cnp.float32_t kl_divergence = 0.0
    cdef long counter

    # Initialize the result matrix
    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.zeros((clusters, particles), dtype=np.float32)
    cdef int bin

    for i in prange(particles, nogil = True, num_threads = num_threads):#, nogil = True, num_threads = num_threads):#particles

        for j in range(clusters):

            kl_divergence = 0.0
            #ignore the rows, that are not present in row_P
            for k in range(sizes[i]):
                
                bin = non_zero_bins[i,k]
                counter = counter + 1
                kl_divergence = kl_divergence + P[bin, i] * (log2P[bin, i] - log2Q[bin, j])

            result[j, i] = kl_divergence

    return result, counter
