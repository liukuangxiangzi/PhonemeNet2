import numpy as np
import time
import h5py

def h5_fast_write(data2save, saving_path, key_name):
    f= h5py.File(saving_path, 'w')
    ds = f.create_dataset(key_name, data=data2save)
    f.close()
    return ds


start = time.time()
y = np.load('data/viseme_techlead.npy') #(33000, 75)
print(y.shape)
end = time.time()
print(end-start)
#y_27 = x[:27000,:]
#y_6 = x[27000:,:]
print('loading index')
#train_index = np.load('data/tmp/train_index.npy')
#test_index = np.load('data/tmp/test_index.npy')

print('indexing training data and test data')
#y_train = y_27[train_index, :]
#y_test = y_27[test_index, :]
#print('train', y_train.shape)
#print('test', y_test .shape)

#h5_fast_write(y_train, 'data/y_train.h5', 'y_train') #(21600, 75)
#h5_fast_write(y_test, 'data/y_test.h5', 'y_test') #(5400, 75)
h5_fast_write(y, 'data/y_techlead.h5', 'y_techlead') #(6000, 75)