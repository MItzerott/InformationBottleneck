import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
import pickle
import time
import copy
import pandas as pd
import IPython
vlog = np.log
vexp = np.exp
from dataset_refined import dataset_refined
from helper_functions import kullback_leibler, entropy
from global_config import load_config
import local_config
config_filename = local_config.config_filename
print_func_names = bool(load_config(filename = config_filename)['ib_code_params']['print_func_names'] == 1)
benchmark_mode = bool(load_config(filename = config_filename)['ib_code_params']['benchmark_mode'] == 1)

import h5py

class model_refined:
    """A representation / interface for an IB model, primarily consisting of
    the encoder / clustering q(t|x) and its associated distributions.
    
    Functions of main interest to users are (in order) __init__, fit,
    report_metrics, report_param, and possibly clamp. The rest are primarily
    helper functions that won't be called directly by the user."""

    def __init__(self,ds,alpha,beta,Tmax=None,qt_x=None,p0=None,waviness=None,
                 ctol_abs=10**-4,ctol_rel=0.,cthresh=1,ptol=10**-10,zeroLtol=0,
                 geoapprox=False,step=None,dt=None,quiet=True):
        """ds is a dataset object (see dataset class above). alpha and beta are
        IB parameters that appear in the generalized cost functional (see
        Strouse & Schwab 2016). Tmax is the maximum number of clusters allowed,
        i.e. the maximum cardinality of T. qt_x is the initialization of the 
        encoder. If not provided, qt_x will be initialized randomly based on
        p0 and waviness (see init_qt_x below for details). ctol_abs, ctol_rel,
        and cthresh are convergence tolerances; see check_converged below for
        details. ptol is the threshold for considering a probability to be zero;
        clusters with probability mass below ptol are pruned. zeroLtol governs
        how aggressively converged solutions are replaced with the single-cluster
        solution; if converged L>zeroLtol, it gets replaced (see
        check_single_better below for details). geoapprox determines whether a
        particular approximation to the IB algorithm is used; applicable to
        geometric datasets where coord is available only. quiet is a flag that
        suppresses some output."""
        if not(isinstance(ds,dataset_refined)):
            raise ValueError('ds must be a dataset')
        self.ds = ds # dataset
        if dt is None: self.dt = self.ds.dt
        else: self.dt = dt
        if alpha<0: raise ValueError('alpha must be a non-negative scalar')
        self.alpha = alpha
        if not(beta>0): raise ValueError('beta must be a positive scalar')
        self.beta = beta
        if Tmax is None:
            Tmax = 10648#ds.X
            if not(quiet): print('Tmax set to %i based on X' % Tmax)
        elif Tmax<1 or Tmax!=int(Tmax):
            raise ValueError('Tmax must be a positive integer')
        elif Tmax>ds.X:            
            print('Reduced Tmax from %i to %i based on X' % (Tmax,ds.X))
            Tmax = ds.X
        else: Tmax = int(Tmax)
        self.Tmax = Tmax
        self.T = Tmax
        if ctol_rel==0 and ctol_abs==0:
            raise ValueError('One of ctol_rel and ctol_abs must be postive')
        if ctol_abs<0 or not(isinstance(ctol_abs,float)):
            raise ValueError('ctol_abs must be a non-negative float')
        self.ctol_abs = ctol_abs
        if ctol_rel<0 or not(isinstance(ctol_rel,float)):
            raise ValueError('ctol_rel must be a non-negative float')
        self.ctol_rel = ctol_rel
        if cthresh<1 or cthresh!=int(cthresh):
            raise ValueError('cthresh must be a positive integer')
        self.cthresh = cthresh
        if not(ptol>0) or not(isinstance(ptol,float)):
            raise ValueError('ptol must be a positive float')
        self.ptol = ptol
        if zeroLtol<0:
            raise ValueError('zeroLtol must be a non-negative float or integer')
        self.zeroLtol = zeroLtol
        self.geoapprox = geoapprox
        self.quiet = quiet
        self.using_labels = False
        self.clamped = False
        self.conv_time = None
        self.conv_condition = None
        self.merged = False
        if step is None: self.step = 0
        if p0 is None:
            if alpha==0: p0 = 1. # DIB default: deterministic init that spreads points evenly across clusters
            else: p0 = 1.#0.75#.75 # non-DIB default: DIB-like init but with only 75% prob mass on "assigned" cluster
        elif p0<-1 or p0>1 or not(isinstance(p0,(int,float))):
            raise ValueError('p0 must be a float/int between -1 and 1')
        else: p0 = float(p0)
        self.p0 = p0
        if waviness is not None and (waviness<0 or waviness>1 or not(isinstance(waviness,float))):
            raise ValueError('waviness must be a float between 0 and 1')
        self.waviness = waviness
        start_time = time.time()
        if qt_x is not None: # use initialization if provided
            if not(isinstance(qt_x,np.ndarray)):
                raise ValueError('qt_x must be a numpy array')
            if isinstance(qt_x,np.ndarray):
                if np.any(qt_x<0) or np.any(qt_x>1):
                    print(qt_x)
                    File = h5py.File('wrong_qt_x.hdf5', mode = 'w')
                    File.create_dataset(name = 'wrong_qt_x', data = qt_x, dtype = self.ds.dt)
                    File.close()
                    raise ValueError('entries of qt_x must be between 0 and 1')
                if qt_x.shape[0]==1: # if single cluster
                    if np.any(qt_x!=1): raise ValueError('columns of qt_x must be normalized')
                #elif np.any(abs(np.sum(qt_x,axis=0)-1)>ptol): # if multi-cluster
                #    raise ValueError('columns of qt_x must be normalized')
            self.qt_x = qt_x.astype(self.dt)
            self.T = qt_x.shape[0]
        else: # initialize randomly if not
            self.init_qt_x()
            
        self.make_step(init=True)
        self.step_time = time.time()-start_time
        if not self.quiet: print('step %i: ' % self.step + self.report_metrics())
    
    def init_qt_x(self):
        """Initializes q(t|x) for generalized Information Bottleneck.
        
        For p0 = 0: init is random noise. If waviness = None, normalized uniform
        random vector. Otherwise, uniform over clusters +- uniform noise of
        magnitude waviness.
        
        For p0 positive: attempt to spread points as evenly across clusters as
        possible. Prob mass p0 is given to the "assigned" clusters, and the
        remaining 1-p0 prob mass is randomly assigned. If waviness = None, again
        use a normalized random vector to assign the remaining mass. Otherwise,
        uniform +- waviness again.
        
        For p0 negative: just as above, except that all data points are "assigned"
        to the same cluster (well, at least |p0| of their prob mass).""" 
        if print_func_names: print('init_qt_x')
        if self.p0==0: # don't insert any peaks; init is all "noise"
            if self.waviness: # flat + wavy style noise
                self.qt_x = np.ones((self.T,self.ds.X))+2*(np.random.rand(self.T,self.ds.X)-.5)*self.waviness # 1+-waviness%                
                for i in range(self.ds.X):
                    self.qt_x[:,i] = self.qt_x[:,i]/np.sum(self.qt_x[:,i]) # normalize
            else: # uniform random vector
                self.qt_x = np.random.rand(self.T,self.ds.X)
                self.qt_x = np.multiply(self.qt_x,np.tile(1./np.sum(self.qt_x,axis=0),(self.T,1))) # renormalize
                if not self.quiet: print('init_qt_x, renormalization: ', np.sum(self.qt_x * self.ds.px), 'should be 1!')
                
        elif self.p0>0: #spread points evenly across clusters; "assigned" clusters for each data point get prob mass p0
            if self.waviness:
                # insert wavy noise part
                self.qt_x = np.ones((self.T,self.ds.X))+2*(np.random.rand(self.T,self.ds.X)-.5)*self.waviness # 1+-waviness%                
                # choose clusters for each x to get spikes
                n = math.ceil(float(self.ds.X)/float(self.T)) # approx number points per cluster
                I = np.repeat(np.arange(0,self.T),n).astype("int") # data-to-cluster assignment vector
                np.random.shuffle(I)
                for i in range(self.ds.X):
                    self.qt_x[I[i],i] = 0 # zero out that cluster
                    self.qt_x[:,i] = (1-self.p0)*self.qt_x[:,i]/np.sum(self.qt_x[:,i]) # normalize others to 1-p0
                    self.qt_x[I[i],i] = self.p0 # insert p0 spike
            else: # uniform random vector instead of wavy
                self.qt_x = np.zeros((self.T,self.ds.X))
                # choose clusters for each x to get spikes
                n = math.ceil(float(self.ds.X)/float(self.T)) # approx number points per cluster
                I = np.repeat(np.arange(0,self.T),n).astype("int") # data-to-cluster assignment vector
                np.random.shuffle(I)
                for i in range(self.ds.X):
                    u = np.random.rand(self.T)
                    u[I[i]] = 0
                    u = (1-self.p0)*u/np.sum(u)
                    u[I[i]] = self.p0
                    self.qt_x[:,i] = u
                
        else: # put all points in the same cluster; primary cluster gets prob mass |p0|
            p0 = -self.p0
            if self.waviness:      
                self.qt_x = np.ones((self.T,self.ds.X))+2*(np.random.rand(self.T,self.ds.X)-.5)*self.waviness # 1+-waviness%
                t = np.random.randint(self.T) # pick cluster to get delta spike
                self.qt_x[t,:] = np.zeros((1,self.ds.X)) # zero out that cluster
                self.qt_x = np.multiply(self.qt_x,np.tile(1./np.sum(self.qt_x,axis=0),(self.T,1))) # normalize the rest...
                self.qt_x = (1-p0)*self.qt_x # ...to 1-p0
                self.qt_x[t,:] = p0*np.ones((1,self.ds.X)) # put in delta spike
            else: # uniform random vector instead of wavy
                self.qt_x = np.zeros((self.T,self.ds.X))
                # choose clusters for each x to get spikes
                t = np.random.randint(self.T) # pick cluster to get delta spike
                for i in range(self.ds.X):
                    u = np.random.rand(self.T)
                    u[t] = 0
                    u = (1-p0)*u/np.sum(u)
                    u[t] = p0
                    self.qt_x[:,i] = u
        if self.qt_x.dtype != self.dt: self.qt_x = self.qt_x.astype(self.dt)
        if not self.quiet: print('Normalization init qt_x: ', np.sum(self.qt_x), np.sum(np.sum(self.qt_x, axis = 0)*self.ds.px))
    
    def qt_step(self):
        """Peforms q(t) update step for generalized Information Bottleneck."""
        if print_func_names: print('performing qt_step')
       
        self.qt = np.dot(self.qt_x,self.ds.px).astype(self.dt)

        dropped = self.qt<=self.ptol # clusters to drop due to near-zero prob
        print(f"Would have dropped {np.sum(~dropped)} clusters", flush = True)
        
        if any(dropped):
            print('Dropped some clusters!', flush = True)
            self.qt = self.qt[~dropped] # drop ununsed clusters
            self.qt_x = self.qt_x[~dropped,:]
            self.T = len(self.qt) # update number of clusters
            self.qt_x = np.multiply(self.qt_x,np.tile(1./np.sum(self.qt_x,axis=0),(self.T,1))) # renormalize          
            self.qt = np.dot(self.qt_x,self.ds.px).astype(self.dt)
            if not self.quiet: print('%i cluster(s) dropped. Down to %i cluster(s).' % (len(np.argwhere(dropped == True)),self.T))
        
        if not self.quiet:
            print('Check if q(t) is normalized: ', np.sum(self.qt))
            print('Check if q(t|x) is normalized: ', np.sum(self.qt_x * self.ds.px))
            
    def qy_t_step(self):
        """Peforms q(y|t) update step for generalized Information Bottleneck."""
        if print_func_names: print('performing qy_t_step')
        self.qy_t = np.dot(self.ds.py_x,np.multiply(self.qt_x,np.outer(1./self.qt,self.ds.px)).T)
        if self.qy_t.dtype != self.dt: self.qy_t = self.qy_t.astype(self.dt)
        #there is no need to normalize here, since it can be proven, that if p(x,y) is normalized, qy_t is too.
        
    def query_coord(self,x,ptol=0):
        """Returns cluster assignment for new data point not in training set."""
        # currently assumes uniform smoothing; needs extended to nearest-neighbor
        if print_func_names: print('performing query_coord')
        if self.alpha!=0: raise ValueError('only implemented for DIB (alpha=0)')
        if self.T==1: return 0
        else:
            # which y correspond to which spatial locations? Ygrid tells us!
            py_x = mvn.pdf(self.ds.Ygrid,mean=x,cov=(self.ds.s**2)*np.eye(2))
            ymask = py_x>ptol
            perc_dropped = 100*(1-np.mean(ymask))
            l = vlog(self.qt)-self.beta*kullback_leibler(py_x[ymask],self.qy_t[ymask,:], benchmark = benchmark_mode)
            return np.argmax(l),perc_dropped

    def qt_x_step(self):
        """Peforms q(t|x) update step for generalized Information Bottleneck."""
        if print_func_names: print('performing qt_x_step')
        if self.T==1: self.qt_x = np.ones((1,self.X),dtype=self.dt)
        else:
            self.qt_x = np.zeros((self.T,self.ds.X),dtype=self.dt)
            l = (vlog(np.expand_dims(self.qt, axis = 1))-self.beta*kullback_leibler(self.ds.py_x,self.qy_t, benchmark = benchmark_mode)).astype(self.dt) # [=] T x 1 # scales like X*Y*T
            if self.alpha==0:
                #this needs work!
                for x in range(self.ds.X):
                    self.qt_x[np.argmax(l[:,x]),x] = 1
            else:
                # exponentials of very small numbers cause underflows.
                # in float64, the lowest number is approx. e^-745
                # for float32, it is approx. e^-103
                # exponentials of very large numbers cause overflows.
                # for float32, it is approx. e^88
                if self.dt == np.float32:
                    if np.min(l/self.alpha) < -103:
                        l[l/self.alpha < -103] = -103 * self.alpha
                        print('Underflow at l/alpha = ', np.min(l/self.alpha), flush = True)
                    if np.max(l/self.alpha) > 88:
                        l[l/self.alpha > 88] = 88 * self.alpha
                        print('Overflow at l/alpha = ', np.max(l/self.alpha), flush = True)
                elif self.dt == np.float64:
                    if np.min(l/self.alpha) < -745:
                        l[l/self.alpha < -745] = -745 * self.alpha
                        print('Underflow at l/alpha = ', np.min(l/self.alpha), flush = True)
                    if np.max(l/self.alpha) > 709:
                        l[l/self.alpha > 709] = 709 * self.alpha
                        print('Overflow at l/alpha = ', np.max(l/self.alpha), flush = True)
                
                #maybe leave things as log for now, normalize and then afterwards turn it into exp form! Maybe we can forego some issues that way
                self.qt_x = vexp(l/self.alpha)/np.sum(vexp(l/self.alpha), axis = 0)
                
                #for later
                # log_qt_x = l/self.alpha - vlog(np.nansum(vexp(l/self.alpha)))
                
        if self.qt_x.dtype != self.dt: self.qt_x = self.qt_x.astype(self.dt)
        if not self.quiet: print('Normalization qt_x: ', np.sum(self.qt_x, axis = 0))

    def build_dist_mat_own(self):
        """Replaces the qy_t_step whens using geoapprox."""
        from scipy.spatial import KDTree
        if print_func_names: print('build_dist_mat_own')
        self.Dxt = np.zeros((self.ds.X,self.T))
        #DxT.shape = (N,N_C)
        cluster_centers = np.zeros((self.T, self.ds.coord.shape[1])) 
        box_size = self.ds.box_size
        if(box_size == None): box_size = float(input('boxsize: '))
        Dist = self.Dxt.copy()
        for cluster in range(self.T):
            index = np.argwhere(self.qt_x[cluster,:] != 0)
            tree = KDTree(self.ds.coord[index].reshape((len(index), 3)), boxsize = self.ds.box_size)
            if(len(index) != 0):
                cluster_center = np.mean(self.ds.coord[index], axis = 0)
                tree_2 = KDTree(cluster_center, boxsize = self.ds.box_size)
                dist = tree.sparse_distance_matrix(tree_2, max_distance = float(box_size)).toarray()
                Dist[index,cluster] = dist

        self.Dxt = Dist

    def qt_x_step_geoapprox_own(self):
        """Peforms q(t|x) update step for approximate generalized Information
        Bottleneck, an algorithm for geometric clustering."""
        if print_func_names: print('performing qt_x_step_geoapprox')
        if self.T==1: self.qt_x = np.ones((1,self.ds.X),dtype=self.dt)
        else:
            self.qt_x = np.zeros((self.T,self.ds.X),dtype=self.dt)
            l = vlog(np.expand_dims(self.qt, axis = 1))-(self.beta/(2*self.ds.s**2))*self.Dxt.T # [=] T x 1 # scales like X*Y*T
            if self.alpha==0: self.qt_x[np.argmax(l),np.arange(self.ds.X)] = 1
            else: self.qt_x = vexp(l/self.alpha)/np.sum(vexp(l/self.alpha)) # note: l/alpha<-745 is where underflow creeps in
        if self.qt_x.dtype != self.dt: self.qt_x = self.qt_x.astype(self.dt)

        self.qt_x /= np.sum(np.sum(self.qt_x, axis = 0) * self.ds.px)
        if not self.quiet: print('Normalization qt_x: ', np.sum(self.qt_x), np.sum(self.qt_x * self.ds.px))

    def calc_metrics(self):
        """Calculates IB performance metrics.."""
        if print_func_names: print('calc_metrics')
        
        self.ht = entropy(self.qt)
        self.hy_t = np.dot(self.qt,entropy(self.qy_t))
        self.iyt = self.ds.hy-self.hy_t
        print("Here are the infos::", self.ds.hy, self.ds.pxy.shape, flush = True)
        self.ht_x = np.dot(self.ds.px,entropy(self.qt_x))
        self.ixt = self.ht-self.ht_x
        self.L = self.ht-self.alpha*self.ht_x-self.beta*self.iyt
        if not self.quiet: print('I(X,Y) = 7.005, ', 'I(Y,C) = ', self.iyt, 'I(X,C) = ', self.ixt)
        
    def report_metrics(self):
        """Returns string of model metrics."""
        if print_func_names: print('report_metrics')
        self.calc_metrics()
        return 'I(X,T) = %.3f, H(T) = %.3f, T = %i, H(X) = %.3f, I(Y,T) = %.3f, I(X,Y) = %.3f, L = %.3f' % (self.ixt,self.ht,self.T,self.ds.hx,self.iyt,self.ds.ixy,self.L)

    def report_param(self):
        """Returns string of model parameters."""
        if print_func_names: print('report_param')
        if self.p0 is None or self.qt_x is not None: p0_str = 'None'
        else: p0_str = '%.3f' % self.p0
        if self.waviness is None or self.qt_x is not None: waviness_str = 'None'
        else: waviness_str = '%.2f' % self.waviness
        if self.ds.smoothing_type is None: smoothing_type_str = 'None'
        else: smoothing_type_str = self.ds.smoothing_type
        if self.ds.smoothing_center is None: smoothing_center_str = 'None'
        else: smoothing_center_str = self.ds.smoothing_center
        if self.ds.s is None: s_str = 'None'
        else: s_str = '%.2f' % self.ds.s
        if self.ds.d is None: d_str = 'None'
        elif self.ds.d==int(self.ds.d): d_str = '%i' % self.ds.d
        else: d_str = '%.1f' % self.ds.d
        return 'alpha = %.2f, beta = %.4f, Tmax = %i, p0 = %s, wav = %s, geo = %s,\nctol_abs = %.0e, ctol_rel = %.0e, cthresh = %i, ptol = %.0e, zeroLtol = %.0e\nsmoothing_type = %s, smoothing_center = %s, s = %s, d = %s' %\
              (self.alpha, self.beta, self.Tmax, p0_str, waviness_str, self.geoapprox,
               self.ctol_abs, self.ctol_rel, self.cthresh, self.ptol, self.zeroLtol,
               smoothing_type_str, smoothing_center_str, s_str, d_str)

    def make_step(self,init=False):
        """Performs one IB step."""
        if print_func_names: print('make_step')

        if not(init):
            start_time = time.time()
            if self.geoapprox: self.qt_x_step_geoapprox_own()
            else: self.qt_x_step()
        self.qt_step()
        self.qy_t_step()
        if self.geoapprox: self.build_dist_mat_own()
        else: self.Dxt = None
        self.calc_metrics()
        self.step += 1
        self.merged = False
        if not(init):
            self.step_time = time.time()-start_time
        if not self.quiet: 
            #check all the datatypes to make sure, they are all the same and all float32
            print(self.qt.dtype, self.qy_t.dtype, self.qt_x.dtype, self.ds.px.dtype, self.ds.pxy.dtype, self.ds.py_x.dtype, self.ds.py.dtype)
              
    def clamp(self):
        """Clamps solution to argmax_t of q(t|x) for each x, i.e. hard clustering."""
        if print_func_names: print('clamp')
        print('before clamp: ' + self.report_metrics())
        if self.alpha==0: print('WARNING: clamping with alpha=0; solution is likely already deterministic.')       
        for x in range(self.ds.X):
            tstar = np.argmax(self.qt_x[:,x])
            self.qt_x[tstar,x] = 1
        self.qt_step()
        self.qy_t_step()
        if self.geoapprox: self.build_dist_mat_own()
        self.clamped = True
        print('after clamp: ' + self.report_metrics())
        
    def panda(self,dist_to_keep=set()):
        """"Return dataframe of model. If dist, include distributions.
        If conv, include converged variables; otherwise include stepwise."""
        if print_func_names: print('panda')
        df = pd.DataFrame(data={
                'alpha': self.alpha, 'beta': self.beta, 'step': self.step,
                'L': self.L, 'ixt': self.ixt, 'iyt': self.iyt, 'ht': self.ht,
                'T': self.T, 'ht_x': self.ht_x, 'hy_t': self.hy_t,
                'hx': self.ds.hx, 'ixy': self.ds.ixy, 'Tmax': self.Tmax,
                'p0': self.p0, 'waviness': self.waviness,  'ptol': self.ptol,
                'ctol_abs': self.ctol_abs, 'ctol_rel': self.ctol_rel,
                'cthresh': self.cthresh, 'zeroLtol': self.zeroLtol,
                'clamped': self.clamped, 'geoapprox': self.geoapprox,
                'using_labels': self.using_labels, 'merged': self.merged,
                'smoothing_type': self.ds.smoothing_type,
                'smoothing_center': self.ds.smoothing_center,
                's': self.ds.s, 'd': self.ds.d,
                'step_time': self.step_time, 'conv_time': self.conv_time,
                'conv_condition': self.conv_condition}, index = [0])
        if 'qt_x' in dist_to_keep:
            df['qt_x'] = [self.qt_x]
        if 'qt' in dist_to_keep:
            df['qt'] = [self.qt]
        if 'qy_t' in dist_to_keep:
            df['qy_t'] = [self.qy_t]
        if 'Dxt' in dist_to_keep:
            df['Dxt'] = [self.Dxt]
        return df

    def depanda(self,df):
        """Replaces current model with one in df."""
        if print_func_names: print('depanda')
        self.alpha = df['alpha'][0]
        self.beta = df['beta'][0]
        self.step = df['step'][0]
        self.L = df['L'][0]
        self.ixt = df['ixt'][0]
        self.iyt = df['iyt'][0]
        self.ht = df['ht'][0]
        self.T = df['T'][0]
        self.ht_x = df['ht_x'][0]
        self.hy_t = df['hy_t'][0]
        self.hx = df['hx'][0]
        self.ixy = df['ixy'][0]
        self.Tmax = df['Tmax'][0]
        self.p0 = df['p0'][0]
        self.waviness = df['waviness'][0]
        self.ptol = df['ptol'][0]
        self.ctol_abs = df['ctol_abs'][0]
        self.ctol_rel = df['ctol_rel'][0]
        self.cthresh = df['cthresh'][0]
        self.zeroLtol = df['zeroLtol'][0]
        self.clamped = df['clamped'][0]
        self.geoapprox = df['geoapprox'][0]
        self.using_labels = df['using_labels'][0]
        self.merged = df['merged'][0]
        self.ds.smoothing_type = df['smoothing_type'][0]
        self.ds.smoothing_center = df['smoothing_center'][0]
        self.ds.s = df['s'][0]
        self.ds.d = df['d'][0]
        self.step_time = df['step_time'][0]
        self.conv_time = df['conv_time'][0]
        self.conv_condition = df['conv_condition'][0]
        self.qt_x = df['qt_x'][0]
        self.qt = df['qt'][0]
        self.qy_t = df['qy_t'][0]
        self.Dxt = df['Dxt'][0]


    def append_conv_condition(self,cond):   
        if print_func_names: print('append_conv_condition')     
        if self.conv_condition is None: self.conv_condition = cond
        else: self.conv_condition += '_AND_' + cond
                
    def update_sw(self):
        """Appends current model / stats to the internal stepwise dataframe."""
        if print_func_names: print('update_sw')
        if self.keep_steps:
            # store stepwise data                
            #self.metrics_sw = self.metrics_sw.append(self.panda(), ignore_index = True)
            self.metrics_sw = pd.concat([self.metrics_sw, self.panda()], ignore_index = True)
            #if bool(self.dist_to_keep): self.dist_sw = self.dist_sw.append(self.panda(self.dist_to_keep), ignore_index = True)
            if bool(self.dist_to_keep): self.dist_sw = pd.concat([self.dist_sw, self.panda(self.dist_to_keep)], ignore_index = True)

    def check_converged(self):
        """Checks if most recent step triggered convergence, and stores step /
        reverts model to last step if necessary."""
        if print_func_names: print('check_converged')
        Lold = self.prev['L'][0] 
        
        # check for small changes
        small_abs_changes = abs(Lold-self.L)<self.ctol_abs
        small_rel_changes = (abs(Lold-self.L)/abs(Lold))<self.ctol_rel
        if small_abs_changes or small_rel_changes: self.cstep += 1
        else: self.cstep = 0 # reset counter of small changes in a row
        if small_abs_changes and self.cstep>=self.cthresh:
            self.conv_condition = 'small_abs_changes'
            print('converged due to small absolute changes in objective', flush = True)  
        if small_rel_changes and self.cstep>=self.cthresh:
            self.append_conv_condition('small_rel_changes')
            print('converged due to small relative changes in objective', flush = True)
            
        # check for objective becoming NaN
        if np.isnan(self.L):
            self.cstep = self.cthresh            
            self.append_conv_condition('cost_func_NaN')
            print('stopped because objective = NaN', flush = True)
            
        L_abs_inc_flag = self.L>(Lold+self.ctol_abs)
        L_rel_inc_flag = self.L>(Lold+(abs(Lold)*self.ctol_rel))
        
        # check for reduction to single cluster
        if self.T==1 and not(L_abs_inc_flag) and not(L_rel_inc_flag):
            self.cstep = self.cthresh
            self.append_conv_condition('single_cluster')
            print('converged due to reduction to single cluster', flush = True)
            
        # check if obj went up by amount above threshold (after 1st step)
        if (L_abs_inc_flag or L_rel_inc_flag) and self.step>1: # if so, don't store or count this step!
            self.cstep = self.cthresh
            if L_abs_inc_flag:
                self.append_conv_condition('cost_func_abs_inc')
                print('converged due to absolute increase in objective value')
            if L_rel_inc_flag:
                self.append_conv_condition('cost_func_rel_inc')
                print('converged due to relative increase in objective value')
            # revert to metrics/distributions from last step
            self.prev.conv_condition = self.conv_condition
            self.depanda(self.prev)
        # otherwise, store step
        else: self.update_sw()
        
        # if converged, check if single cluster solution better
        # commented out to check for behaviour
        if self.cstep>=self.cthresh and self.T>1: self.check_single_better()
                
    def check_single_better(self):
        """ Replace converged step with single-cluster map if better."""
        if print_func_names: print('check_single_better')
        sqt_x = np.zeros((self.T,self.ds.X),dtype=self.dt)
        sqt_x[0,:] = 1.
        smodel = model_refined(ds=self.ds,alpha=self.alpha,beta=self.beta,Tmax=self.Tmax,
                       qt_x=sqt_x,p0=self.p0,waviness=self.waviness,
                       ctol_abs=self.ctol_abs,ctol_rel=self.ctol_rel,cthresh=self.cthresh,
                       ptol=self.ptol,zeroLtol=self.zeroLtol,geoapprox=self.geoapprox,
                       quiet=True)
        smodel.step = self.step
        smodel.conv_condition = self.conv_condition + '_AND_force_single'
        if smodel.L<(self.L-self.zeroLtol): # if better fit...
            print("single-cluster mapping reduces L from %.4f to %.4f (zeroLtol = %.1e); replacing solution." % (self.L,smodel.L,self.zeroLtol))
            # replace everything
            self.depanda(smodel.panda(dist_to_keep={'qt_x','qt','qy_t','Dxt'}))
            self.update_sw()
            print('single-cluster solution: ' + self.report_metrics())            
        else: print("single-cluster mapping not better; changes L from %.4f to %.4f (zeroLtol = %.1e)." % (self.L,smodel.L,self.zeroLtol)) 
        
    def check_merged_better(self,findbest=True):
        """Checks if merging any two clusters improves cost function.
        
        If findbest = True, review all merges and choose best, if any improve L.
        If findbest = False, just accept the first merge that improves L.
        Latter option good if too many clusters to compare all."""
        if print_func_names: print('check_merged_better')
        start_time = time.time()
        anybetter = False
        best = copy.deepcopy(self)
        # iterate over cluster pairs
        for t1 in range(self.T-1):
            for t2 in range(t1+1,self.T):
                if not(anybetter) or findbest:
                    # copy model
                    alt = copy.deepcopy(self)
                    alt.quiet = True
                    # t2 -> t1
                    alt.qt_x[t1,alt.qt_x[t2,:]==1] = 1
                    alt.qt_x[t2,:] = 0
                    # update other dist
                    alt.make_step(init=True)
                    # check if cost function L reduced relative to best so far
                    if alt.L<best.L:
                        best = copy.deepcopy(alt)
                        mergedt1 = t1
                        mergedt2 = t2
                        anybetter = True
        if anybetter:
            print('merged clusters %i and %i, reducing L from %.3f to %.3f' % (mergedt1,mergedt2,self.L,best.L))
            self.__init__(ds=self.ds,alpha=self.alpha,beta=self.beta,
                          Tmax=self.Tmax,qt_x=best.qt_x,p0=self.p0,waviness=self.waviness,
                          ctol_abs=self.ctol_abs,ctol_rel=self.ctol_rel,cthresh=self.cthresh,
                          ptol=self.ptol,zeroLtol=self.zeroLtol,geoapprox=self.geoapprox,
                          step=self.step,quiet=self.quiet)
            self.merged = True
            self.step_time = time.time()-start_time
            self.cstep = 0
            self.conv_condition = None
            self.update_sw()
            return True
        else:
            self.merged = False
            print('no merges reduce L')
            return False

    def fit(self,keep_steps=False,dist_to_keep={'qt_x','qt','qy_t','Dxt'}, benchmark_mode = benchmark_mode):
        """Runs generalized IB algorithm to convergence for current model.
        keep_steps determines whether pre-convergence models / statistics about
        them are kept. dist_to_keep is a set with the model distributions to be
        kept for each step."""
        if print_func_names: print('fit')
        fit_start_time = time.time()
                       
        self.keep_steps = keep_steps
        self.dist_to_keep = dist_to_keep
        
        print(20*'*'+' Beginning IB fit with the following parameters '+20*'*')
        print(self.report_param())
        print(88*'*', flush = True)
        
        # initialize stepwise dataframes, if tracking them
        if self.keep_steps:
            self.metrics_sw = self.panda()
            if bool(self.dist_to_keep): self.dist_sw = self.panda(self.dist_to_keep)
        
        # check if single cluster init
        if self.T==1:
            self.cstep = self.cthresh
            print('converged due to initialization with single cluster')
            self.conv_condition = 'single_cluster_init'
        else: # init iterative parameters
            self.cstep = 0
            self.conv_condition = None
            self.conv_condition = None
        
        # save encoder init
        self.qt_x0 = self.qt_x
        print('Summed qt_x: ', np.sum(self.qt_x * self.ds.px), flush = True)
        
        # benchmark test requires only one step for KL calculations.
        if benchmark_mode:
            self.prev = self.panda(dist_to_keep={'qt_x','qt','qy_t','Dxt'}) 
            self.make_step()
            if not(self.quiet): print('step %i: ' % self.step + self.report_metrics(), 'beta = %.2f, alpha = %.2f' %(self.beta, self.alpha))
            self.check_converged()
            if self.cstep>=self.cthresh and self.T>1 and self.alpha==0: self.check_merged_better()
        else:
            # iterate to convergence
            while self.cstep<self.cthresh:
                self.prev = self.panda(dist_to_keep={'qt_x','qt','qy_t','Dxt'}) 
                self.make_step()
                if not(self.quiet): print('step %i: ' % self.step + self.report_metrics(), 'beta = %.2f, alpha = %.2f' %(self.beta, self.alpha))
                self.check_converged()
                if self.cstep>=self.cthresh and self.T>1 and self.alpha==0: self.check_merged_better()

        # report
        print('converged in %i step(s) to: ' % self.step + self.report_metrics(), flush = True)
        
        # clean up
        self.cstep = None
        self.prev = None
        self.step_time = None

        # record total time to convergence
        self.conv_time = time.time() - fit_start_time
        
    def plot_qt_x(self):
        """Visualizes clustering induced by q(t|x), if coord available."""
        if print_func_names: print('plot_qt_x')
        if self.ds.coord is None:
            raise ValueError('coordinates not available; cannot plot')        
        #if not(self.alpha==0 or self.clamped):
        #   raise ValueError('qt_x not determinstic; cannot plot')
         
        # build cluster assignment vector
        cluster = np.zeros(self.ds.X)
        for x in range(self.ds.X): cluster[x] = np.argmax(self.qt_x[:,x])
            
        # plot with ggplot
        plt.figure()
        plt.scatter(self.ds.coord[:,0],self.ds.coord[:,1],c=cluster)
        plt.title('found %i clusters' %len(np.unique(cluster)))
        plt.show()
