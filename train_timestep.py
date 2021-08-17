

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



def h5_fast_read(reading_path, key_name):
    f= h5py.File(reading_path, 'r')
    ds = f[key_name][:]
    # f.close()
    return ds

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


    model.save('model/audio2pho_model_ep200_1e-4_32_34sub_50khz_timestep15.h5')
    print("Saved model to disk")


run_model()







