
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential, save_model, load_model
import h5py
import numpy as np
from tqdm import tqdm
from train_window import DataGenerator


def h5_fast_read(reading_path, key_name):
    f= h5py.File(reading_path, 'r')
    ds = f[key_name][:]
    # f.close()
    return ds

# path_x_vali = 'data/X_LJ.h5'
# path_y_vali = 'data/y_LJ.h5'
#
# x_vali = h5_fast_read(path_x_vali, 'data')
# print(x_vali.shape)
# y_vali = h5_fast_read(path_y_vali, 'y_LJ')
# y_vali = to_categorical(y_vali)
# print(y_vali)

def lstm_data_transform_x(x_data, num_steps):

    """ Changes data to the format for LSTM training
for sliding window approach """

    # Prepare the list for the transformed data
    X= list()
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

        # Append the list with sequencies
        X.append(seq_X)

    # Make final arrays
    x_array = np.array(X)

    print('transformed_x shape', x_array.shape)

    return x_array

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

trainX = np.load('test_data/s21_bbad2n/audio_test_s2_bbad2n.npy')
timesteps = 50

trainY = np.load('test_data/s21_bbad2n/viseme_test_s2_bbad2n.npy')
trainY = to_categorical(trainY)
print('trainy', trainY)
# trainX = trainX.reshape(trainX.shape[0]*trainX.shape[1],  int(trainX.shape[2]))
# trainY = trainY.reshape(trainY.shape[0]*trainY.shape[1], trainY.shape[2])
trainX  = trainX[:, :50, :]
trainY = trainY[:, :50, :]

print(trainY.shape, trainX.shape)


# trainX = lstm_data_transform_x(trainX,timesteps)
# trainX,trainY = lstm_data_transform(trainX,trainY,timesteps)
model = load_model('model/audio2pho_model_ep200_1e-4_32_34_50khz_winsize50.h5') # #audio2pho_model_ep300_1e-4_32_33sub.h5

#results = model.predict(trainX)
results = model.evaluate(trainX, trainY)
print("test loss, test acc:", results)
#print(results.shape)
#print(np.argmax(results,axis=2))

#ep300: [0.3196505904197693, 0.9066666960716248]
#ep200: [0.30473992228507996, 0.9066666960716248]