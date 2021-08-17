import h5py
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm



def h5_fast_write(data2save, saving_path, key_name):
    f= h5py.File(saving_path, 'w')
    ds = f.create_dataset(key_name, data=data2save)
    f.close()
    print( 'saved h5')
    return ds

def h5_fast_read(reading_path, key_name):
    f= h5py.File(reading_path, 'r')
    ds = f[key_name][:]

    # f.close()
    return ds

def lstm_data_transform(x_data, y_data, num_steps):

    """ Changes data to the format for LSTM training
for sliding window approach """

    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop of the entire data set
    for i in tqdm(range(x_data.shape[0])):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequency for y
        seq_y = y_data[end_ix]
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    print('transformed_x shape', x_array.shape)
    print('transformed y shape',y_array.shape)
    return x_array, y_array


def labelcategorical():
    label_train = h5_fast_read('data/y_34_50_train.h5', 'y_34_50_train')
    #label_train = np.array(np.load(args.train_y))  #train_phoneme_label14H.npy
    label_test = h5_fast_read('data/y_34_50_test.h5', 'y_34_50_test')

    # label_train = label_train.reshape(label_train.shape[0]*label_train.shape[1])
    # print(label_train)
    print(label_train.shape)
    #label_test = np.array(np.load(args.test_Y))

    #label_train = label_train.reshape(753,75,-1)
    #label_test = label_test.reshape(191,75,-1)

    # one hot encode y
    label_categorical_train = to_categorical(label_train)
    label_categorical_test = to_categorical(label_test)
    return label_categorical_train, label_categorical_test

def timestep_data(is_training_data, timesteps):
    trainy, testy = labelcategorical()

    if is_training_data:
        trainX = h5_fast_read('data/x_34_50_train.h5', 'x_34_50_train')
        #trainX = np.load('x_train.h5')  #trainX.shape = (753, 75, 768)
        print('trainX.shape', trainX.shape)
        trainX = trainX.reshape(trainX.shape[0]*trainX.shape[1],  int(trainX.shape[2]))
        trainy = trainy.reshape(trainy.shape[0]*trainy.shape[1], trainy.shape[2])

        trainX,trainy = lstm_data_transform(trainX,trainy,timesteps)
        h5_fast_write(trainX,'data/timestep_data/x_34_50_train_timestep5.h5', 'x_34_50_train_timestep5')
        h5_fast_write(trainy,'data/timestep_data/y_34_50_train_timestep5.h5', 'y_34_50_train_timestep5')
    else:
        testX = h5_fast_read('data/x_34_50_test.h5', 'x_34_50_test')
        #testX = np.load(args.test_X)   #testX.shape = (191, 75, 768)
        print('testX.shape', testX.shape)
        testX = testX.reshape(testX.shape[0]*testX.shape[1],  int(testX.shape[2]))
        testy = testy.reshape(testy.shape[0] * testy.shape[1], int(testy.shape[2]))


        testX,testy = lstm_data_transform(testX,testy,timesteps)
        h5_fast_write(testX,'data/timestep_data/x_34_50_test_timestep5.h5', 'x_34_50_test_timestep5')
        h5_fast_write(testy,'data/timestep_data/y_34_50_test_timestep5.h5', 'y_34_50_test_timestep5')



timestep_data(False,5)