import numpy as np
import os
import tables
import h5py
import time
import random


def convert2hdf(pytables=False, h5=False):
    if pytables:
        x_all = np.load('data/.npy')
        f = tables.open_file('data/.hdf', 'w')
        atom = tables.Atom.from_dtype(x_all.dtype)
        ds = f.create_carray(f.root, 'x_all', atom, x_all.shape)
        ds[:] = x_all
        f.close()
    if h5:
        x_all = np.load('data/s_testtechlead_48_3.npy')
        f= h5py.File('data/s_testtechlead_48_3.h5', 'w')
        ds = f.create_dataset('data', data=x_all)
        f.close()










convert2hdf(h5=True)
