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

def entropy_parallel(cnp.ndarray[cnp.float32_t, ndim=2] P, cnp.ndarray[cnp.float32_t, ndim=2] log2P, int num_threads):
    cdef int xdim = P.shape[0]
    cdef int ydim = P.shape[1]
    cdef int i, k
    cdef cnp.ndarray[cnp.float32_t, ndim=1] entropy = np.zeros(ydim, dtype=np.float32)
    cdef cnp.float32_t entr = 0.0

    for i in prange(ydim, nogil = True, num_threads = num_threads):
        entr = 0.0
        for k in range(xdim):
            entr = entr - (P[k, i] * log2P[k, i])
        entropy[i] = entr
    
    return entropy
