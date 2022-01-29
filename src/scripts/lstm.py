import numpy as np
import pandas as pd
import random

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU
from tqdm import tqdm

from data_handler_lstm import RESULT_COLS

RANDOM_SEED = 42
OPTIMIZER = 'adam'
LOSS = 'mse'

'''
Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language.
Schematically, a RNN layer uses a for loop to iterate over the timesteps of a sequence, while maintaining an internal state that encodes information 
about the timesteps it has seen so far.
Long Short-Term Memory networks, or LSTMs for short, can be applied to time series forecasting.
'''
class AlgoLSTM:
    def __init__(self, data_handler_lstm, num_of_epochs):
        # Initialize GPU
        gpu_devices = tf.config.list_physical_devices('GPU')
        for device in gpu_devices:
            #print(device)
            tf.config.experimental.set_memory_growth(device, True)
        # setup random seeds
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        self.data_handler_lstm = data_handler_lstm
        window_size = data_handler_lstm.window_size
        n_features = len(data_handler_lstm.features)

        self.model = model = Sequential()
        model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(window_size, n_features)))
        model.add(LSTM(64, activation='tanh', input_shape=(window_size, 128)))
        model.add(Dense(16))
        model.add(LeakyReLU())
        model.add(Dense(4))
        model.add(LeakyReLU())
        model.add(Dense(1)) # 1 target feature only
        model.compile(optimizer=OPTIMIZER, loss=LOSS)

        train_gen = data_handler_lstm.train_gen
        val_gen = data_handler_lstm.val_gen

        model.fit(train_gen, validation_data = val_gen, epochs = num_of_epochs, verbose=1)

    def get_forecasts(self):
        model = self.model
        train_gen = self.data_handler_lstm.train_gen
        val_gen = self.data_handler_lstm.val_gen
        test_gen = self.data_handler_lstm.test_gen

        results = {}
        for col in RESULT_COLS:
            results[col] = []

        # predicting train
        day_counter = 0
        for i in tqdm(range(len(train_gen))):
            x, y = train_gen[i] # x is 1, 2, 6 y is (1,1); 2 is the window size
            yhat = model.predict(x, verbose=0)
            results[RESULT_COLS[0]] += [y[0][0]]     # train_actual
            results[RESULT_COLS[1]] += [yhat[0][0]]  # train_pred
            results[RESULT_COLS[2]] += [day_counter] # train_x_axis
            day_counter += 1

        # predicting validation
        for i in tqdm(range(len(val_gen))):
            x, y = val_gen[i]
            yhat = model.predict(x, verbose=0)
            results[RESULT_COLS[3]] += [y[0][0]]     # val actual
            results[RESULT_COLS[4]] += [yhat[0][0]]  # val pred
            results[RESULT_COLS[5]] += [day_counter] # val_x_axis
            day_counter += 1

        # predicting test
        for i in tqdm(range(len(test_gen))):
            x, y = test_gen[i]
            yhat = model.predict(x, verbose=0)
            results[RESULT_COLS[6]] += [y[0][0]]     # test actual
            results[RESULT_COLS[7]] += [yhat[0][0]]  # test pred
            results[RESULT_COLS[8]] += [day_counter] # test_x_axis
            day_counter += 1

        # print(len(results['test_pred'])) # loose window_size
        # print((~np.isnan(results['test_pred'])).sum())

        # keys of the dictionary become index using orient='index'
        df_result = pd.DataFrame.from_dict(results, orient='index').T
        df_result = df_result.astype({RESULT_COLS[2]: 'int32'}, copy=False) # change to int from float

        df_train = df_result[[RESULT_COLS[0], RESULT_COLS[1], RESULT_COLS[2]]].set_index(RESULT_COLS[2])
        df_train.index.name = 'x'

        df_val = df_result[[RESULT_COLS[3], RESULT_COLS[4], RESULT_COLS[5]]].set_index(RESULT_COLS[5]).dropna(axis=0)
        df_val.index = df_val.index.astype('int32', copy=False)
        df_val.index.name = 'x'

        df_test = df_result[[RESULT_COLS[6], RESULT_COLS[7], RESULT_COLS[8]]].set_index(RESULT_COLS[8])
        last_day = results[RESULT_COLS[8]][-1]
        assert(np.isnan(df_test.loc[last_day][RESULT_COLS[6]]))
        df_test.loc[last_day][RESULT_COLS[6]] = 0 # temporary
        df_test.dropna(inplace=True)
        df_test.loc[last_day][RESULT_COLS[6]] = np.nan # change it back
        df_test.index = df_test.index.astype('int32', copy=False)
        df_test.index.name = 'x'

        df_concatenated = pd.concat([df_train,df_val,df_test], axis=1) # gather the predicted train, val, test dataframes

        return df_concatenated # columns: train_actual, train_pred, train_x_axis,
                               #          val_actual, val_pred, val_x_axis,
                               #          test_actual, test_pred, test_x_axis