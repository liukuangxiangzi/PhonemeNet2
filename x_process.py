import numpy as np
import os
import tables
import h5py
import time
import random

def concatenate_data():
    audio_result_folder_path = 'data/all_audio_result50/'
    subs = os.listdir(audio_result_folder_path)
    subs.sort()
    if os.path.exists(audio_result_folder_path + '.DS_Store') is True:
        subs.remove('.DS_Store')
    c_sub = 0
    audio_feature = np.empty([0, 75, 768])
    for sub in subs:  #33
        c_sub += 1
        print(c_sub)
        sub_dir = audio_result_folder_path + sub + '/'
        sub_name = os.listdir(sub_dir)
        print(sub_dir)
        print(sub_name)
        audio_feature_a_sub = np.load(sub_dir + sub_name[0]) # (1000, 75, 768)
        print('shape1', audio_feature_a_sub.shape)
        audio_feature = np.append(audio_feature, audio_feature_a_sub, axis=0)
    print('shape2', audio_feature.shape)
    np.save(audio_result_folder_path  + 'X_33_50.npy', audio_feature)

def convert2hdf(pytables=False, h5=False):
    if pytables:
        x_all = np.load('data/all_audio_result50/X_34_50.npy')
        f = tables.open_file('data/all_audio_result50/X_34_50.hdf', 'w')
        atom = tables.Atom.from_dtype(x_all.dtype)
        ds = f.create_carray(f.root, 'x_all', atom, x_all.shape)
        ds[:] = x_all
        f.close()
    if h5:
        x_all = np.load('data/all_audio_result50/X_33_50.npy') #(33000, 75, 768)
        f= h5py.File('data/all_audio_result50/X_33_50.h5', 'w')
        ds = f.create_dataset('data', data=x_all)
        f.close()

def make_index():
    random.seed(10)
    train_index = sorted(random.sample(range(34000), int(34000*0.8))) #27000
    test_index = list(set([*range(34000)]) - set(train_index))
    np.save('train_index_3450.npy', train_index)
    np.save('test_index_3450.npy', test_index)




def foo():
    random_num = random.sample(range(10),5)
    list = [*range(10)]
    rest = [a for a in list if a not in random_num]
    print(rest)

def h5_fast_write(data2save, saving_path, key_name):
    f= h5py.File(saving_path, 'w')
    ds = f.create_dataset(key_name, data=data2save)
    f.close()
    return ds

def h5_fast_read(reading_path, key_name):
    f= h5py.File(reading_path, 'r')
    ds = f[key_name]
    # f.close()
    return ds



def divide_data():
    start = time.time()
    x = h5_fast_read('data/X_34_50.h5', 'data')
    print(x.shape)
    end = time.time()
    print(end-start)
    #x_27 = x[:27000,:,:]
    #x_6 = x[27000:,:,:]
    print('loading index')
    train_index = np.load('train_index_3450.npy')
    test_index = sorted(np.load('test_index_3450.npy'))
    print('indexing training data and test data')
    #x_train = x_27[train_index, :, :]
    #x_test = x_27[test_index, :, :]
    x_train = x[train_index, :, :]
    x_test = x[test_index, :, :]

    h5_fast_write(x_train, 'data/x_34_50_train.h5', 'x_34_50_train')
    h5_fast_write(x_test, 'data/x_34_50_test.h5', 'x_34_50_test')







#concatenate_data()
#convert2hdf(h5=True)
#make_index()
divide_data()
# foo()