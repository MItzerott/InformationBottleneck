import numpy as np
from helper_functions import entropy
import os
from scipy.ndimage import gaussian_filter
import pickle
from global_config import load_config
import local_config
config_filename = local_config.config_filename

print_func_names = bool(load_config(filename = config_filename)['ib_code_params']['print_func_names'] == 1)

class dataset_refined:
    """A representation / interface for an IB dataset, primarily consisting of
    the joint distribution p(x,y) and its marginals, conditionals, and stats.
    Also includes functionality for acceping data point coordinates and applying
    smoothing to yield an IB-appropriate p(x,y)."""
    
    def __init__(self,
                 pxy=None,
                 coord=None, # 2d coordinates for x
                 labels=None, # labels y
                 gen_param=None, # generative parameters
                 name=None,
                 smoothing_type='uniform',
                 smoothing_center='data_point',
                 s=None, # smoothing scale
                 d=None, # neighborhood size
                 dt=None, # data type
                 box_size=None, #box_size for periodic BC
                 total_bins=None,
                 quiet=None
                 ): 

        if dt is None: 
            if load_config(filename = config_filename)['ib_code_params']['data_type'] == 32:
                self.dt = np.float32
            elif load_config(filename = config_filename)['ib_code_params']['data_type'] == 64:
                self.dt = np.float64
        else: self.dt = dt
        if total_bins is not None: self.total_bins = total_bins
        else:
            raise ValueError('Total number of bins must be specified!')
        if box_size is not None: self.box_size = box_size
        else:
            raise ValueError('Boxsize must be set for periodic BC!')
        if pxy is not None:
            if not(isinstance(pxy,np.ndarray)):
                raise ValueError('pxy must be a numpy array')
            if np.any(pxy<0) or np.any(pxy>1):
                raise ValueError('entries of pxy must be between 0 and 1')
            if abs(np.sum(pxy)-1)>10**-8:
                raise ValueError('pxy must be normalized; sum = %f' % np.sum(pxy))
            pxy = pxy.astype(self.dt)
        self.pxy = pxy # the distribution that (D)IB acts upon
        if coord is not None:
            if not(isinstance(coord,np.ndarray)): raise ValueError('coord must be a numpy array')
            else: coord = coord.astype(self.dt)
        self.coord = coord # locations of data points if geometric, assumed 2D
        if labels is not None and len(labels)!=coord.shape[0]:
                raise ValueError('number of labels must match number of rows in coord')
        self.labels = labels # class labels of data (if synthetic)
        if smoothing_type in ['u','uniform']: self.smoothing_type = 'uniform'
        elif smoothing_type in ['t','topological']: self.smoothing_type = 'topological'
        elif smoothing_type in ['m','metric']: self.smoothing_type = 'metric'
        elif smoothing_type is not None: raise ValueError('invalid smoothing_type')
        else: self.smoothing_type = None
        if smoothing_center in ['d','data_point']: self.smoothing_center = 'data_point'
        elif smoothing_center in ['m','neighborhood_mean']: self.smoothing_center = 'neighborhood_mean'
        elif smoothing_center in ['b','blended']: self.smoothing_center = 'blended'
        elif smoothing_center is not None: raise ValueError('invalid smoothing_center')
        else: self.smoothing_center = None
        if s is not None and s<0:
            raise ValueError('s must be a positive scalar')
        self.s = s # determines width of gaussian smoothing for coord->pxy
        if d is not None and d<0:
            raise ValueError('d must be a positive scalar')
        self.d = d # determines neighborhood of a data point in gaussian smoothing for coord->pxy
        if gen_param is not None:
            if not(isinstance(gen_param,dict)):
                raise ValueError('gen_param must be a dictionary')
        self.gen_param = gen_param # generative parameters of data (if synthetic)
        self.name = name # name of dataset, used for saving
        if self.pxy is not None:
            self.process_pxy()
            #print('Datatype after process = ', self.pxy.dtype)

        elif self.coord is not None:
            self.X = self.coord.shape[0]
        if self.pxy is None and self.coord is not None and self.s is not None:
            self.coord_to_pxy()

        
    def __str__(self):
        return(self.name)
    
    def process_pxy(self,drop_zeros=True):
        """Drops unused x and y, and calculates info-theoretic stats of pxy."""
        if print_func_names: print('process_pxy')
        Xorig, Yorig = self.pxy.shape
        px = self.pxy.sum(axis=1)
        py = self.pxy.sum(axis=0)
        if drop_zeros:
            nzx = px>0#1e-12 #0 # find nonzero-prob entries
            nzy = py>0#1e-12 #0 #changed from 0 to 1e-12
            zx = np.where(px<=0)[0]
            zy = np.where(py<=0)[0]
            self.px = px[nzx] # drop zero-prob entries
            self.py = py[nzy]
            self.Ygrid = self.Ygrid[nzy]
            pxy_orig = self.pxy
            tmp = pxy_orig[nzx,:]
            self.pxy = tmp[:,nzy] # pxy_orig with zero-prob x,y removed
        else:
            self.px = px
            self.py = py
        self.X = len(self.px)
        self.Y = len(self.py)
        if (Xorig-self.X)>0:
            print('%i of %i Xs dropped due to zero prob; size now %i. Dropped IDs:' % (Xorig-self.X,Xorig,self.X))
            print(zx, flush = True)
        if (Yorig-self.Y)>0:
            print('%i of %i Ys dropped due to zero prob; size now %i. Dropped IDs:' % (Yorig-self.Y,Yorig,self.Y))
            print(zy, flush = True)
        
        if drop_zeros:
            #old way, takes longer, but does not produce significantly different result.
            #py_x_1 = np.multiply(self.pxy.T,np.tile(1./self.px,(self.Y,1)))
            print(np.sum(self.pxy))
            temp_py_x = self.py_x.copy()
            self.py_x = self.pxy.T/self.px
            
            print(np.sum(self.py_x - temp_py_x), flush = True) #around 1e-13, so blah

        self.hx = entropy(self.px)
        self.hy = entropy(self.py)
        self.hy_x = np.dot(self.px,entropy(self.py_x))
        self.ixy = self.hy-self.hy_x
        
    def normalize_coord(self):
        if print_func_names: print('normalize_coord')
        desired_r = 20
    
        min_x1 = np.min(self.coord[:,0])
        min_x2 = np.min(self.coord[:,1])
        max_x1 = np.max(self.coord[:,0])
        max_x2 = np.max(self.coord[:,1])
        range_x1 = max_x1-min_x1
        range_x2 = max_x2-min_x2
        r = (range_x1+range_x2)/2
        
        # zero-mean
        self.coord = self.coord - np.mean(self.coord,axis=0)
        
        # scale
        self.coord = desired_r*self.coord/r
        
    def make_bins(self,pad=None):
        """Compute appropriate spatial bins."""
        if print_func_names: print('make_bins')
        if pad is None: pad = 2*self.s # bins further than this from all data points are dropped
        self.pad = pad
        total_bins = self.total_bins**(self.coord.shape[1])
        # dimensional preprocessing
        min_x1 = np.min(self.coord[:,0])
        min_x2 = np.min(self.coord[:,1])
        min_x3 = np.min(self.coord[:,2])
        max_x1 = np.max(self.coord[:,0])
        max_x2 = np.max(self.coord[:,1])
        max_x3 = np.max(self.coord[:,2])
        range_x1 = max_x1-min_x1
        range_x2 = max_x2-min_x2
        range_x3 = max_x3-min_x3
        bins1 = int((total_bins*range_x1/np.sqrt(range_x2*range_x3))**(1/3)) # divy up bins according to spread of data
        bins2 = int((total_bins*range_x2/np.sqrt(range_x1*range_x3))**(1/3))
        bins3 = int((total_bins*range_x3/np.sqrt(range_x1*range_x2))**(1/3))
        Y = int(bins1*bins2*bins3)
        # generate bins
        min_y1 = min_x1-pad
        max_y1 = max_x1+pad
        min_y2 = min_x2-pad
        max_y2 = max_x2+pad
        min_y3 = min_x3-pad
        max_y3 = max_x3+pad
        y1 = np.linspace(min_y1,max_y1,bins1,dtype=self.dt)
        y2 = np.linspace(min_y2,max_y2,bins2,dtype=self.dt)
        y3 = np.linspace(min_y3,max_y3,bins3,dtype=self.dt)
        y1v,y2v,y3v = np.meshgrid(y3,y2,y1)
        Ygrid = np.array([np.reshape(y1v,Y),np.reshape(y2v,Y),np.reshape(y3v,Y)]).T
        
        return Y,bins1,bins2,bins3,y1v,y2v,y3v,Ygrid
        

    def coord_to_pxy(self,pad=None,drop_distant=False,
                     drop_zeros=True,make_smoothed_coord_density=False, plot_binning = False):
        """Uses smoothing paramters to transform coord into pxy."""
        # assumes 2D coord, total_bins is approximate
        #now also done for 3D coords
        if print_func_names: print('coord_to_pxy')
        if self.smoothing_type is None: raise ValueError('smoothing_type not yet set')
        if self.s is None: raise ValueError('smoothing scale, s, not yet set')      
        if self.smoothing_type=='uniform':
            pass
            #print('smoothing coordinates own: smoothing_type = uniform, scale s = %.2f' % self.s)
        else:
            if self.smoothing_center is None: raise ValueError('smoothing_center not yet set')
            if self.d is None: raise ValueError('neighborhood size, d, not yet set')
            if self.smoothing_type=='topological': 
                print('Smoothing coordinates: smoothing_type = topological, smoothing_center = %s, scale s = %.2f, neighborhood size d = %i' % (self.smoothing_center,self.s,self.d))
            elif self.smoothing_type=='metric': 
                print('Smoothing coordinates: smoothing_type = metric, smoothing_center = %s, scale s = %.2f, neighborhood size d = %.1f' % (self.smoothing_center,self.s,self.d))
            else: raise ValueError('invalid smoothing_type')
        import time
        # compute appropriate spatial bins
        Y,bins1,bins2,bins3,y1v,y2v,y3v,Ygrid = self.make_bins(pad=pad)        
        # construct gaussian-smoothed p(y|x), based on smoothing parameters
        py_x = np.zeros((Y,self.X),dtype=self.dt)
        smoothed_coord_density = np.zeros(y1v.shape,dtype=self.dt)
        ycountv = np.zeros(Y) # counts data points within pad of each bin
        ycount = np.zeros(y1v.shape)
        
        readfromhdf5 = False#True
        writetohdf5 = False
        
        Filename = 'py_x_%i_%i.h5' %(load_config(filename = config_filename)['sim_params']['sim_size'], load_config(filename = config_filename)['ib_model_params']['total_bins_1D'])
        
        if readfromhdf5 and Filename in os.listdir('.'):
            import h5py
            print('Reading from File...')
            h5f = h5py.File(Filename,'r')
            py_x = h5f['py_x'][:].astype(self.dt)
            h5f.close()
        
        else:
            pxy_hist, edges = np.histogramdd(np.array((np.arange(self.X), self.coord[:,0], self.coord[:,1], self.coord[:,2]), dtype = self.dt).T, bins = [self.X, bins1, bins2, bins3])
            #we need to also take into account, that the box has periodic boundary conditions, thus, we need to run the smoothing kernel and the histogram on a larger box
            pxy_hist = pxy_hist.astype(self.dt)
            kernel_width = (self.s**2) # the width of each Gaussian kernel. In PIXEL units!!! (each pixel has a side length: 2*bin_range/Nbins) 

            py_x = gaussian_filter(pxy_hist, kernel_width, axes = [1,2,3], mode = 'wrap')
            del pxy_hist
            
            py_x = np.reshape(py_x, ((self.X, Y))).T
            
        if plot_binning:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(12, 8), layout='constrained')
            
            axes_coord = [fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)]
            axes_down = [fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)]
            
            axes_coord[0].scatter(self.coord[:,1], self.coord[:,0], s = .1, c = 'black')
            axes_coord[0].set_xlabel('Y'); axes_coord[0].set_ylabel('-X')
            axes_coord[1].scatter(self.coord[:,2], self.coord[:,0], s = .2, c = 'black')
            axes_coord[1].set_xlabel('Z'); axes_coord[1].set_ylabel('-X')
            axes_coord[2].scatter(self.coord[:,2], self.coord[:,1], s = .1, c = 'black')
            axes_coord[2].set_xlabel('Z'); axes_coord[2].set_ylabel('-Y')
                    
            axes_down[0].imshow(np.sum(np.sum(py_x, axis = 0)[:,:,:], axis = 2))
            axes_down[1].imshow(np.sum(np.sum(py_x, axis = 0)[:,:,:], axis = 1))
            axes_down[2].imshow(np.sum(np.sum(py_x, axis = 0)[:,:,:], axis = 0))
            
            axes_down[0].set_xlabel('X'); axes_down[0].set_ylabel('Y')
            axes_down[1].set_xlabel('Z'); axes_down[1].set_ylabel('X')
            axes_down[2].set_xlabel('Y'); axes_down[2].set_ylabel('Z')
            
            for ax in axes_coord:
                ax.set_xlim(0, 50000)
                ax.set_ylim(50000, 0)
            #plt.show()
            fig.savefig('Binning_%i_s%.1f.pdf' %(int(self.total_bins**(1/3)), self.s))
        
        if writetohdf5 and not readfromhdf5:

            if Filename in os.listdir('.'):
                overwrite = bool(input('File already exists! Overwrite?[0, 1]') == '1')
                print(overwrite)

            if (not Filename in os.listdir('.')) or (overwrite):
                import h5py
                h5f = h5py.File('py_x_%i_%i.h5' %(load_config(filename = config_filename)['sim_params']['sim_size'], load_config(filename = config_filename)['ib_model_params']['total_bins_1D']), 'w')
                h5f.create_dataset('py_x', data=py_x, dtype = self.dt)
                
                h5f.close()
                            
        if drop_distant:
            # drop ybins that are too far away from data
            ymask = ycountv>0 #ymask is all bins, where the number of element is larger than 0.                
            py_x = py_x[ymask,:]
            Y = np.sum(ymask)
            Ygrid = Ygrid[ymask,:]
            self.bins_dropped = ycount==0
            
        else: self.bins_dropped = None
        self.Y = Y
        self.Ygrid = Ygrid
        
        # normalize p(y|x), since gaussian binned/truncated and bins dropped
        # old way, slower than new way
        # for x in range(self.X): py_x[:,x] = py_x[:,x]/np.sum(py_x[:,x])
        
        py_x /= np.sum(np.sum(py_x, axis = 0) * 1/self.X)
        self.py_x = py_x
        del py_x
        
        # package stuff for plotting smoothed density in coord space
        if make_smoothed_coord_density: self.smoothed_coord_density = smoothed_coord_density/np.sum(smoothed_coord_density[:])
        self.y1v = y1v
        self.y2v = y2v
        
        # construct p(x) and p(x,y)
        self.px = (1/self.X)*np.ones(self.X,dtype=self.dt)
        self.pxy = np.multiply(np.tile(self.px,(self.Y,1)),self.py_x).T

        # calc and display I(x,y)
        self.process_pxy(drop_zeros=False)
        
    def plot_coord(self,save=False,path=None):
        if print_func_names: print('plot_coord')
        if self.coord is not None:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Coordinates plotted')
            plt.scatter(self.coord[:,0],self.coord[:,1], s = 5, marker = '.')
            plt.axis('scaled')
            #plt.show()
            if save:
                fig.savefig('/home/mitzerott/MasterThesis/Analysis/' + self.name+'_coord.pdf',bbox_inches='tight')

        else:
            print("coord not yet defined")
    
    def plot_coord_3D(self,save=False,path=None):
        if print_func_names: print('plot_coord')
        if self.coord is not None:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter3D(self.coord[:500,0], self.coord[:500,1], self.coord[:500,2], color = 'orange')
            ax.scatter3D(self.coord[500:,0], self.coord[500:,1], self.coord[500:,2], color = 'blue')

            if save:
                if path is None: raise ValueError('must specify path to save figure')
                else: fig.savefig(path+self.name+'_coord.pdf',bbox_inches='tight')
        else:
            print("coord not yet defined")
            
    def plot_smoothed_coord(self,save=False,path=None):
        if print_func_names: print('plot_smoothed_coord')
        from matplotlib import pyplot as plt
        fig = plt.figure()
        plt.title('s = %i' % self.s,fontsize=18,fontweight='bold')
        plt.contour(self.y1v,self.y2v,self.smoothed_coord_density)
        plt.scatter(self.coord[:,0],self.coord[:,1])
        #plt.axis('scaled')
        plt.axis([-22,22,-15,15])
        plt.show()
        if save:
            if path is None: raise ValueError('must specify path to save figure')
            else: fig.savefig(path+self.name+'_smoothed_coord_s%i'%self.s+'.pdf',bbox_inches='tight')
        
    def plot_pxy(self,save=False,path=None):
        if print_func_names: print('plot_pxy')
        from matplotlib import pyplot as plt
        fig = plt.figure()
        if self.pxy is not None:
            plt.ylabel('X',fontsize=14,fontweight='bold')
            plt.xlabel('Y',fontsize=14,fontweight='bold')
            plt.imshow(self.pxy, cmap = 'turbo', norm = 'log')
            #plt.contourf(self.pxy)
            plt.colorbar()
            plt.title('s = %.1f' %self.s)

            if save:
                fig.savefig(self.name+'_pxy_s%i'%self.s+'.png',bbox_inches='tight')
                #if path is None: raise ValueError('must specify path to save figure')
                #else: fig.savefig(path+self.name+'_pxy_s%i'%self.s+'.pdf',bbox_inches='tight')
        else:
            print("pxy not yet defined")
            
    def save(self,directory,filename=None):
        """Pickles dataset in directory with filename."""
        if print_func_names: print('save')
        if filename is None: filename = self.name+'_dataset'
        with open(directory+filename+'.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            
    def load(self,directory,filename=None):
        """Replaces current content with pickled data in directory with filename."""
        if print_func_names: print('load')
        if filename is None: filename = self.name+'_dataset'
        with open(directory+filename+'.pkl', 'rb') as input:
            obj = pickle.load(input)
        self.__init__(pxy = obj.pxy, coord = obj.coord, labels = obj.labels,
                      gen_param = obj.gen_param, name = obj.name, s = obj.s)
  
