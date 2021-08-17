import numpy as np
import time
import h5py

def h5_fast_write(data2save, saving_path, key_name):
    f= h5py.File(saving_path, 'w')
    ds = f.create_dataset(key_name, data=data2save)
    f.close()
    return ds


start = time.time()
y = np.load('data/viseme_34_50.npy') #(34000, 75) #, allow_pickle=True
print('y', y.shape)
end = time.time()
print(end-start)
#y_27 = y[:27000,:]
#y_6 = y[27000:,:]
print('loading index')
train_index = np.load('train_index_3450.npy')
test_index = sorted(np.load('test_index_3450.npy'))

print('indexing training data and test data')

y_train = y[train_index, :]
print(y_train)
y_test = y[test_index, :]
print('train', y_train.shape, y_train.dtype)
print('test', y_test .shape)


h5_fast_write(y_train, 'data/y_34_50_train.h5', 'y_34_50_train') #(21600, 75)
h5_fast_write(y_test, 'data/y_34_50_test.h5', 'y_34_50_test') #(5400, 75)

#h5_fast_write(y, 'data/y_LJ.h5', 'y_LJ') #(5400, 75)