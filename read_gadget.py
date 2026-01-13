import h5py
import numpy as np
from os.path import exists

def find_files(filepath):
    nfiles = 0
    fname = filepath + '.{}.hdf5'.format(nfiles)
    if not exists(fname):
        raise ValueError('File does not exist: {}'.format(fname))

    while exists(fname):
        nfiles += 1
        fname = filepath + '.{}.hdf5'.format(nfiles)
    return nfiles

def read_array(filepath, group):
    nfiles = find_files(filepath)

    # get data dimensions
    arr_lens = np.zeros(nfiles+1, dtype=np.int64)
    for i in range(nfiles):
        fname = filepath + '.{}.hdf5'.format(i)
        with h5py.File(fname, 'r') as f:
            try:
                dset = f[group]
            except KeyError:
                print('Warning: {} not found in {}'.format(group,fname))
                continue
            arr_shape = np.array(dset.shape)
            arr_lens[i+1] = dset.shape[0] + arr_lens[i]
    arr_shape[0] = arr_lens[-1]

    # load data into numpy array
    arr = np.zeros(arr_shape)
    for i in range(nfiles):
        fname = filepath + '.{}.hdf5'.format(i)
        with h5py.File(fname, 'r') as f:
            try:
                dset = f[group]
            except KeyError:
                print('Warning: {} not found in {}'.format(group,fname))
                continue
            arr[arr_lens[i]:arr_lens[i+1]] = dset

    return arr

def read_attribute(filepath, attribute):
    fname = filepath + '.0.hdf5'
    if not exists(fname):
        raise ValueError('File does not exist: {}'.format(fname))

    with h5py.File(fname, 'r') as f:
        header = f['Header']
        attr = header.attrs[attribute]
    return attr



if __name__ == '__main__':

    filepath = '/store/erebos/spfeifer/CLUES/RUNS/L100N128/output/groups_004/fof_subhalo_tab_004'

    h = read_attribute(filepath, 'HubbleParam')
    print('HubbleParam :', h)
    a = read_attribute(filepath, 'Time')
    print('Time :', a)
    z = read_attribute(filepath, 'Redshift')
    print('Redshift :', z)

    pos = read_array(filepath, 'Group/GroupPos')
    print('FOF Coordinates :\n', pos)
    mass = read_array(filepath, 'Group/Group_M_Crit200')
    print('FOF M_200crit :\n', mass)
    vel = read_array(filepath, 'Group/GroupVel')
    print('FOF Velocities :\n', vel)
