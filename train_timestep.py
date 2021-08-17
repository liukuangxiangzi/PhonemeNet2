
#prprocess Y

import os
import h5py
from datetime import datetime
import numpy as np

from keras.layers import Input, LSTM
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import *
from keras.utils import plot_model, Sequence
from tqdm import tqdm
import argparse


#python train.py -x path-of-audio-feature-file-for-training, -X path-of-audio-feature-file-for-testing, -y path-of-phoneme-file-for-training -Y path-of-phoneme-file-for-testing -o path-of-saving-model
#e.g. python train.py -x data/audio_feature_train.npy -X data/audio_feature_test.npy -y data/train_phoneme_label_13.npy -Y data/test_phoneme_label_13.npy -o model/audio2pho_model_mfa13label_ep300_1e-4_32.h5

def h5_fast_read(reading_path, key_name):
    f= h5py.File(reading_path, 'r')
    ds = f[key_name][:]
    # f.close()
    return ds




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


# load the dataset, returns train and test X and y elements
def load_dataset(transform = False, timesteps=5):
    # load all x
    trainX = h5_fast_read('data/x_34_50_train.h5', 'x_34_50_train')
    #trainX = np.load('x_train.h5')  #trainX.shape = (753, 75, 768)
    print('trainX.shape', trainX.shape)
    testX = h5_fast_read('data/x_34_50_test.h5', 'x_34_50_test')
    #testX = np.load(args.test_X)   #testX.shape = (191, 75, 768)
    print('testX.shape', testX.shape)
    # load all y
    trainy, testy = labelcategorical()
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)

    if transform:
        trainX = trainX.reshape(trainX.shape[0]*trainX.shape[1],  int(trainX.shape[2]))
        testX = testX.reshape(testX.shape[0]*testX.shape[1],  int(testX.shape[2]))
        trainy = trainy.reshape(trainy.shape[0]*trainy.shape[1], trainy.shape[2])
        testy = testy.reshape(testy.shape[0] * testy.shape[1], int(testy.shape[2]))
        testX,testy = lstm_data_transform(testX,testy,timesteps)
        trainX,trainy = lstm_data_transform(trainX,trainy,timesteps)



    return trainX, trainy, testX, testy

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


class DataGenerator(Sequence) :

    def __init__(self, audio_data, labels, batch_size) :
        self.audio_data = audio_data
        self.labels = labels
        self.batch_size = batch_size


    def __len__(self) :
        return (np.ceil(len(self.audio_data) / float(self.batch_size))).astype(np.int)


    def __getitem__(self, idx) :
        batch_x = self.audio_data[idx * self.batch_size : (idx+1) * self.batch_size, :, :]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size,:]

        return batch_x,batch_y

def run_model():
    #set tensorboard
    #tensorboard --logdir /Users/liukuangxiangzi/PycharmProjects/PhonemeNet/logs/fit/ --host=127.0.0.1
    log_dir = os.path.join(
        "logs",
        "fit",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tbCallBack = TensorBoard(log_dir= log_dir, histogram_freq=0, write_graph=True, write_images=True)

    #X y data
    # trainX, trainy, testX, testy = load_dataset(transform = True, timesteps = 5)
    trainX = h5_fast_read('data/timestep_data/x_34_50_train_timestep5.h5', 'x_34_50_train_timestep5')
    trainy = h5_fast_read('data/timestep_data/y_34_50_train_timestep5.h5', 'y_34_50_train_timestep5')

    print('training data loaded',trainy.shape)
    testX = h5_fast_read('data/timestep_data/x_34_50_test_timestep5.h5', 'x_34_50_test_timestep5')
    testy = h5_fast_read('data/timestep_data/y_34_50_test_timestep5.h5', 'y_34_50_test_timestep5')
    print('test data loaded', testy.shape)

    verbose, epochs, batch_size = 0, 50, 32

    training_data_generator = DataGenerator(trainX,trainy,batch_size)
    test_data_generator = DataGenerator(testX,testy,batch_size)

    h_dim = 256
    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    print(n_timesteps, n_features)

    drpRate = 0.2
    recDrpRate = 0.2
    lr = 1e-4
    initializer = 'glorot_uniform'

    # define model
    net_in = Input(shape=(n_timesteps, n_features))
    lstm1 = LSTM(h_dim,
                 activation='sigmoid',
                 dropout=drpRate,
                 recurrent_dropout=recDrpRate,
                 return_sequences=True)(net_in)
    lstm2 = LSTM(h_dim,
                 activation='sigmoid',
                 dropout=drpRate,
                 recurrent_dropout=recDrpRate,
                 return_sequences=True)(lstm1)
    lstm3= LSTM(h_dim,
                activation='sigmoid',
                dropout=drpRate,
                recurrent_dropout=recDrpRate,
                return_sequences=False)(lstm2)

    dropout = Dropout(0.5)(lstm3)

    l1 = Dense(128,
               kernel_initializer=initializer,
               name='lm_Dense1',activation='relu')(dropout)

    out = Dense(13,
                kernel_initializer=initializer, name='lm_out',activation='softmax')(l1)

    model = Model(inputs=net_in, outputs=out)
    model.summary()
    opt = Adam(lr=lr)
    #opt = SGD(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])



    plot_model(model, to_file='audio2phoneme_model.png', show_shapes=True, show_layer_names=True)

    model.fit(x=training_data_generator,
              validation_data=test_data_generator,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[tbCallBack],
              shuffle=False

              )


    model.save('model/audio2pho_model_ep200_1e-4_32_34sub_50khz_ts15.h5')
    print("Saved model to disk")


run_model()







