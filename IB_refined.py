import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal as mvn
import time
import math
import pickle
import copy
vlog = np.vectorize(math.log)
vexp = np.vectorize(math.exp)
from model_refined import model_refined as model
from dataset_refined import dataset_refined
from global_config import load_config
import local_config
config_filename = local_config.config_filename
print_func_names = bool(load_config(filename = config_filename)['ib_code_params']['print_func_names'] == 1)

# A word on notation: for probability variables, an underscore here means a
# conditioning, so read _ as |.

def output_data_to_file(metrics_conv, metrics_sw, dist_conv, beta, dist_sw, Filename = 'test.hdf5'):
    import h5py

    this_beta = str(beta).split('.')
    metrics_conv.to_hdf(Filename, key = 'metrics_conv', mode = 'a')
    metrics_sw.to_hdf(Filename, key = 'metrics_sw', mode = 'a')
    
    q_distr_conv = [dist_conv['qt_x'].to_numpy()][0]

    Filename_t_x = Filename.split('.hdf5')[0] + '_t_x.hdf5'
    
    # It is possible to store all distributions in files, but this is generally not adviced for large datasets
    # Filename_t = Filename.split('.hdf5')[0] + '_t.hdf5'
    # Filename_y_t = Filename.split('.hdf5')[0] + '_y_t.hdf5'
    
    with h5py.File(Filename_t_x, 'a') as hf_t_x:
        for i, dataset0 in enumerate(q_distr_conv):
            hf_t_x.create_dataset('qt_x_conv_iteration_%i_%i_%i' %(i, int(this_beta[0]), int(this_beta[1])), data=dataset0)

    # store other distributions as well
    '''
    with h5py.File(Filename_t, 'a') as hf_t:
        for i, dataset1 in enumerate(q_distr_conv[1]):
            hf_t.create_dataset('qt_conv_iteration_%i_%i_%i' %(i, int(this_beta[0]), int(this_beta[1])), data=dataset1)

    with h5py.File(Filename_y_t, 'a') as hf_y_t:
        for i, dataset2 in enumerate(q_distr_conv[2]):
            hf_y_t.create_dataset('qy_t_conv_iteration_%i_%i_%i' %(i, int(this_beta[0]), int(this_beta[1])), data=dataset2)
    '''
    return 0
    
def set_param(fit_param,param_name,def_val):
    """Helper function for IB.py to handle setting of fit parameters."""
    if print_func_names: print('set_param')
    param = fit_param.get(param_name,def_val) # extract param, def = def_val
    if param is None: param = def_val # if extracted param is None, use def_val
    return param

def make_param_dict(fit_param,*args):
    """Helper function for IB.py to handle setting of keyword arguments for the
    IB_single.py function."""
    if print_func_names: print('make_param_dict')
    param_dict = {}
    for arg in args: # loop over column names passed
        if fit_param.get(arg,None) is not None: # if col specified exists and isn't None...
            param_dict[arg] = fit_param[arg] # ...add to keyword dictionary
    return param_dict

def IB_refined(ds,fit_param,paramID=None,conv_dist_to_keep={'qt_x','qt','qy_t','Dxt'},
       keep_steps=True,sw_dist_to_keep={'qt_x','qt','qy_t','Dxt'}, quiet = True, output = False):
    """Performs many generalized IB fits to a single dataset. One series of
    fits (across beta) is performed for each row of input dataframe fit_param.
    Columns correspond to fit parameters.
    
    INPUTS
    ds is a dataset object; see class definition above.
    fit_param = pandas df, with each row specifying a single round of IB fits,
            where a "round" means a bunch of fits where the only parameter that
            varies from fit to fit is beta; columns include:
        0) alpha = IB parameter interpolating between IB and DIB (required)
        *** parameters with defaults set below ***
        1) betas = list of initial beta values to run, where beta is an IB parameter
            specifying the "coarse-grainedness" of the solution
        2) beta_search = boolean indicating whether to perform automatic beta search
            or use *only* initial beta(s)
        3) max_fits = max number of beta fits allowed for each input row
        4) max_time = max time (in seconds) allowed for fitting of each input row
        5) repeats = repeated fits per beta / row, after which the fit with best value
            of objective function L is retained
        6) clamp = boolean indicating whether to clamp fit after convergence;
            both unclamped and clamped model are stored and returned
        *** parameters passed to IB model object (defaults and documentation there) ***    
        7-13) Tmax, p0, waviness, ctol_abs, ctol_rel, cthresh, ptol, zeroLtol

    
    OUTPUTS
    all 4 outputs are pandas dfs. "sw" means stepwise; each row corresponds to a
    step in the IB algorithm. "conv" means converged; each row corresponds to a
    converged solution of the IB algorithm. "metrics" has columns corresponding
    to things like the objective function value L, informations and entropies,
    and the number of clusters used, while "dist" has columns for the actual
    distributions being optimizing, such as the encoder q(t|x). thus, dist dfs
    are much larger than metrics dfs.
    
    most column values should be self-explanatory, or are explained above."""
    if print_func_names: print('IB_own')
    # set defaults

    def_betas = list(
        np.logspace(
            start = load_config(filename = config_filename)['ib_fit_params']['beta_start'], 
            stop = load_config(filename = config_filename)['ib_fit_params']['beta_end'], 
            num = load_config(filename = config_filename)['ib_fit_params']['beta_steps'], 
            base = 2)
        )
    def_betas.sort(reverse = bool(load_config(filename = config_filename)['ib_fit_params']['reverse_sweep'] != 1))

    # initialize primary dataframes
    
    metrics_conv = None
    metrics_sw = None
    dist_conv = None
    dist_sw = None
    
    # iterate over fit parameters (besides beta, which is done below)
    fit_param = fit_param.where((pd.notnull(fit_param)), None) # NaN -> None
    if paramID is None: paramID = 0 # all betas have same
    else: paramID -= 1 # since it'll get ticked below
    fitID = 0 # all repeats have same
    fitIDwrep = 0 # repeats get unique
    # note that for clamped/unclamped version of a model, all 3 IDs are same
    
    
    #the code can do runs for multiple parameters, including how long the code should run, how many repeats and so on
    #currently, only one fit_param dictionary is used at a given time, which means, that the outermost loop only runs once.
    for irow in range(len(fit_param.index)):
        # tick counter
        paramID += 1
        
        # extract parameters for this fit
        this_fit = fit_param.iloc[irow]
        this_alpha = this_fit['alpha']
        
        # parameters that have defaults set above
        this_betas = def_betas[:] # slice here to pass by value, not ref
        this_beta_search = this_fit['beta_search']
        this_max_fits = this_fit['max_fits']
        this_max_time = this_fit['max_time']
        this_repeats = int(this_fit['repeats'])
        this_geoapprox = this_fit['geoapprox']
        this_clamp = this_fit['clamp']
        this_using_labels = this_fit['using_labels']
        
        # optional parameters that have defaults set by IB_single.py
        param_dict = make_param_dict(this_fit,'Tmax','p0','waviness','ctol_abs','ctol_rel','cthresh','ptol','zeroLtol')
        
        this_ds = ds
        
        # make pre-fitting initializations
        betas = this_betas.copy() # stack of betas
        fit_count = 0
        fit_time = 0
        fit_start_time = time.time()
        
        # loop over betas
        while fit_count<=this_max_fits and fit_time<=this_max_time and len(betas)>0:
            
            # tick counter
            fitID += 1
            
            # pop beta from stack
            this_beta = betas.pop()
            
            # init data structures that will store the repeated fits for this particular setting of parameters
            these_metrics_sw = None
            these_metrics_conv = None
            these_dist_sw = None
            these_dist_conv = None
            
            init_random = bool(load_config(filename = config_filename)['ib_fit_params']['init_random'] == 1)
            #save the current qt_x, such that the model can take the prior of the previous model
                    
            # loop over repeats                                        
            for repeat in range(this_repeats):
                
                # tick counter
                fitIDwrep += 1
                if not(quiet): print('+'*15+' repeat %i of %i '%(repeat+1,this_repeats)+'+'*15)
                
                # do a single fit and extract resulting dataframes
                # if this is the first time, the code runs, take random initial clustering,
                # otherwise, use the previously converged to distributions as a prior.

                if init_random:
                    m = model(ds=this_ds,alpha=this_alpha,beta=this_beta,**param_dict,geoapprox=False)
                if not init_random and fit_count == 0:
                    m = model(ds=this_ds,alpha=this_alpha,beta=this_beta,**param_dict,geoapprox=False, qt_x = None)
                if not init_random and fit_count != 0:
                    #add some small perturbation to previous qt_x!
                    
                    # if all previous iterations failed/are nan, then there is no prev_qt_x
                    if prev_qt_x is None:
                        perturbed_qt_x = None
                    else:
                        print('Perturbed qt_x')
                        np.random.seed()
                        noise = np.random.normal(
                            scale=load_config(filename = config_filename)['ib_fit_params']['perturbation'], 
                            size=prev_qt_x.shape
                            )
                                        
                        # Add noise and clip to avoid negative values
                        perturbed_qt_x = np.clip(prev_qt_x + noise, 1e-12, 1)
                        
                        del noise
                        del prev_qt_x
                        
                        #and normalize again afterwards
                        perturbed_qt_x = np.multiply(perturbed_qt_x,np.tile(1./np.sum(perturbed_qt_x,axis=0),(perturbed_qt_x.shape[0],1)))
                        #print(np.min(prev_qt_x), np.max(prev_qt_x))
                        
                        if np.any(perturbed_qt_x<0) or np.any(perturbed_qt_x>1):
                            print(
                                np.min(these_dist_conv.loc[these_dist_conv['repeat'] == best_repeat, 'qt_x'].values[0]), 
                                np.max(these_dist_conv.loc[these_dist_conv['repeat'] == best_repeat, 'qt_x'].values[0])
                                )
                            print(np.min(perturbed_qt_x), np.max(perturbed_qt_x))
                            print('Normalization qt_x_prev: ', np.sum(perturbed_qt_x, axis = 0))                    
                        if not(quiet): print('Normalization qt_x_prev: ', np.sum(perturbed_qt_x, axis = 0))
                        
                    m = model(ds=this_ds,alpha=this_alpha,beta=this_beta,**param_dict,geoapprox=False, qt_x = perturbed_qt_x)

                if output:
                    Filename = 'output_hpc_%i_10%s_initrandom_%i_pert_%s_reverse_%s_rep_%i_betas_%i_beta_%f_rev_%i_bins_%i_alpha_%.1f_p0_%.1f_wav_%s_%i.hdf5' %(
                        load_config(filename = config_filename)['sim_params']['sim_size'],
                        load_config(filename = config_filename)['sim_params']['sim_identifier'],
                        load_config(filename = config_filename)['ib_fit_params']['init_random'],
                        str(load_config(filename = config_filename)['ib_fit_params']['perturbation']),
                        str(load_config(filename = config_filename)['ib_fit_params']['reverse_sweep']),
                        load_config(filename = config_filename)['ib_fit_params']['repeats'],
                        load_config(filename = config_filename)['ib_fit_params']['beta_steps'],
                        this_beta, 
                        load_config(filename = config_filename)['ib_fit_params']['reverse_sweep'], 
                        load_config(filename = config_filename)['ib_model_params']['total_bins_1D'], 
                        this_alpha,
                        m.p0,
                        str(m.waviness),
                        np.random.randint(low = 1, high= 90, size = 1)
                    )
            
                m.fit(keep_steps=keep_steps,dist_to_keep=sw_dist_to_keep)
                    
                this_metrics_conv = m.panda()
                if bool(conv_dist_to_keep): this_dist_conv = m.panda(conv_dist_to_keep)
                
                # extract sw models as necessary
                if keep_steps:
                    this_metrics_sw = m.metrics_sw
                    if bool(sw_dist_to_keep): this_dist_sw = m.dist_sw
                
                # once converged and sw models extracted, clamp if necessary
                if this_clamp and this_alpha!=0:
                    m.clamp()
                    #this_metrics_conv = this_metrics_conv.append(m.panda(), ignore_index = True)
                    this_metrics_conv = pd.concat([this_metrics_conv, m.panda()], ignore_index = True)
                    #if bool(conv_dist_to_keep): this_dist_conv = this_dist_conv.append(m.panda(conv_dist_to_keep), ignore_index = True)
                    if bool(conv_dist_to_keep): this_dist_conv = pd.concat([this_dist_conv, m.panda(conv_dist_to_keep)], ignore_index = True)
                    
                # if also running geoapprox...
                if this_geoapprox:
                    
                    # ...then run with approx on same init
                    m2 = model(ds=this_ds,alpha=this_alpha,beta=this_beta,**param_dict,geoapprox=True,qt_x=m.qt_x0)
                    m2.fit(keep_steps=keep_steps,dist_to_keep=sw_dist_to_keep)
                    
                    # ...and append results
                    this_metrics_conv = pd.concat([this_metrics_conv, m2.panda()], ignore_index = True)
                    if bool(conv_dist_to_keep): this_dist_conv = pd.concat([this_dist_conv, m2.panda(conv_dist_to_keep)], ignore_index = True)
                    
                    # extract sw models as necessary
                    if keep_steps:
                        this_metrics_sw = pd.concat([this_metrics_sw, m2.metrics_sw], ignore_index = True)
                        if bool(sw_dist_to_keep): this_dist_sw = pd.concat([this_dist_sw, m2.dist_sw], ignore_index = True)
                    
                    # once converged and sw models extracted, clamp if necessary
                    if this_clamp and this_alpha!=0:
                        m2.clamp()
                        this_metrics_conv = pd.concat([this_metrics_conv, m2.panda()], ignore_index = True)
                        if bool(conv_dist_to_keep): this_dist_conv = pd.concat([this_dist_conv, m2.panda(conv_dist_to_keep)], ignore_index = True)
                        
                # add repeat labels and fit IDs, which are specific to this repeat            
                this_metrics_conv['repeat'] = repeat
                this_metrics_conv['fitIDwrep'] = fitIDwrep
                if keep_steps:
                    this_metrics_sw['repeat'] = repeat
                    this_metrics_sw['fitIDwrep'] = fitIDwrep
                if bool(conv_dist_to_keep):
                    this_dist_conv['repeat'] = repeat
                    this_dist_conv['fitIDwrep'] = fitIDwrep
                if bool(sw_dist_to_keep):
                    this_dist_sw['repeat'] = repeat
                    this_dist_sw['fitIDwrep'] = fitIDwrep
                    
                # add this repeat to these repeats
                if these_metrics_conv is not None: these_metrics_conv = pd.concat([these_metrics_conv, this_metrics_conv], ignore_index = True)
                else: these_metrics_conv = this_metrics_conv
                if keep_steps:
                    if these_metrics_sw is not None: these_metrics_sw = pd.concat([these_metrics_sw, this_metrics_sw], ignore_index = True)
                    else: these_metrics_sw = this_metrics_sw
                if bool(conv_dist_to_keep):
                    if these_dist_conv is not None: these_dist_conv = pd.concat([these_dist_conv, this_dist_conv], ignore_index = True)
                    else: these_dist_conv = this_dist_conv
                if bool(sw_dist_to_keep):
                    if these_dist_sw is not None: these_dist_sw = pd.concat([these_dist_sw, this_dist_sw], ignore_index = True)
                    else: these_dist_sw = this_dist_sw
                    
            # end of repeat fit loop for single beta
            if not(quiet): print('+'*15+' finished these repeats '+'+'*15)
             
            # add number of repeats and fit ID (without repeats)
            these_metrics_conv['paramID'] = paramID # this is assigned earlier than expected to help with beta refinement
            these_metrics_conv['fitID'] = fitID
            these_metrics_conv['repeats'] = this_repeats
            if keep_steps:
                these_metrics_sw['paramID'] = paramID
                these_metrics_sw['fitID'] = fitID
                these_metrics_sw['repeats'] = this_repeats
            if bool(conv_dist_to_keep):
                these_dist_conv['paramID'] = paramID
                these_dist_conv['fitID'] = fitID
                these_dist_conv['repeats'] = this_repeats
            if bool(sw_dist_to_keep):
                these_dist_sw['paramID'] = paramID
                these_dist_sw['fitID'] = fitID
                these_dist_sw['repeats'] = this_repeats
            
            # mark best repeat (lowest L): ignored clamped, approx, and true init fits
            df = these_metrics_conv[(these_metrics_conv['clamped']==False) & (these_metrics_conv['geoapprox']==False) & (these_metrics_conv['using_labels']==False)]
            #print(df)
            best_id = df['L'].idxmin()
            if np.isnan(best_id): # if all repeats NaNs, just use first repeat
                best_repeat = 0
            else: # otherwise use best
                best_repeat = these_metrics_conv['repeat'].loc[best_id]
            these_metrics_conv['bestrep'] = False
            these_metrics_conv.loc[these_metrics_conv['repeat'] == best_repeat, 'bestrep'] = True
            if keep_steps:
                these_metrics_sw['bestrep'] = False
                these_metrics_sw.loc[these_metrics_sw['repeat'] == best_repeat, 'bestrep'] = True
            if bool(conv_dist_to_keep):
                these_dist_conv['bestrep'] = False
                these_dist_conv.loc[these_dist_conv['repeat'] == best_repeat, 'bestrep'] = True
            if bool(sw_dist_to_keep):
                these_dist_sw['bestrep'] = False
                these_dist_sw.loc[these_dist_sw['repeat'] == best_repeat, 'bestrep'] = True
                
            # advance fit counters
            fit_count += this_repeats
            fit_time = time.time()-fit_start_time
            
            if output:
                if sw_dist_to_keep != None:
                    output_data_to_file(these_metrics_conv, these_metrics_sw, these_dist_conv, this_beta, these_dist_sw, Filename = Filename)
                else:
                    output_data_to_file(these_metrics_conv, these_metrics_sw, this_beta, these_dist_conv, Filename = Filename)
            
            #sometimes, the objective function is nan, because a run failed. Therefore only take the previous step converged solution, if it is not a failed run.
            #print(these_metrics_conv.loc[these_metrics_conv['repeat'] == best_repeat, 'L'].values)
            if (not init_random) and (m.T != 1):
                if np.isnan(these_metrics_conv.loc[these_metrics_conv['repeat'] == best_repeat, 'L'].values) == False:
                    prev_qt_x = these_dist_conv.loc[these_dist_conv['repeat'] == best_repeat, 'qt_x'].values[0]
                    
                else:
                    prev_qt_x = None
            else:
                prev_qt_x = None
            
            display_variables = True
            if display_variables:
                import sys
                def sizeof_fmt(num, suffix='B'):
                    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
                    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
                        if abs(num) < 1024.0:
                            return "%3.1f %s%s" % (num, unit, suffix)
                        num /= 1024.0
                    return "%.1f %s%s" % (num, 'Yi', suffix)
                
                for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                                        locals().items())), key= lambda x: -x[1])[:10]:
                    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
                            
            these_metrics_conv = None
            these_metrics_sw = None
            these_dist_conv = None
            these_dist_sw = None
            
        if fit_count>=this_max_fits: print('Stopped beta refinement because ran over max fit count of %i' % this_max_fits)
        if fit_time>=this_max_time: print('Stopped beta refinement because ran over max fit time of %i seconds' % this_max_time)
        if len(betas)==0: print('Beta refinement complete.')
        #m.plot_qt_x()
        metrics_conv = None
        metrics_sw = None
        dist_conv = None
        dist_sw = None
        
    # end iteration over fit parameters
    return metrics_conv, dist_conv, metrics_sw, dist_sw
