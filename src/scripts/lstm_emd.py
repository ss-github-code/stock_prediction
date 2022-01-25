import numpy as np
import pandas as pd
import random
import tensorflow as tf

from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, LeakyReLU
from scipy.interpolate import CubicSpline

from data_handler_lstm import FEATURES, RESULT_COLS, WIN_SIZE_FOR_IMFS, SKIP_REMAIN_FOR_IMFS

# CONSTANTS
MODEL_EPOCHS = {
    'IMF1': 10,
    'IMF2': 10,
    'IMF3': 3,
    'IMF4': 3,
    'IMF5': 3,
    'IMF6': 2,
    'IMF7': 2,
    'IMF8': 1,
    'Rsd':  1,
    'DEFAULT': 1,
}
RANDOM_SEED = 42
OPTIMIZER = 'adam'
LOSS = 'mse'
IMFS_TO_PREDICT_USING_LSTM = ['IMF1','IMF2']

class SplineModel(): # used for low frequency IMF components (other than IMFS_TO_PREDICT_USING_LSTM)
    def __init__(self,time_series_generator):
        self.name = "SplineModel"
        self.gen = time_series_generator
    
    def predict(self, x_window, verbose=0):
        result = []
        x_window = np.squeeze(x_window, axis=0)
        last_element_index = x_window.shape[1]-1
        series = x_window[:,last_element_index].reshape(-1)
        cs = CubicSpline(np.arange(len(series)), series)
        next_value = cs(len(series)+1)
        result += [next_value]

        return np.array(result).reshape(1,-1) # 1,-1

class AlgoLSTM_EMD:
    def __init__(self, data_handler_lstm):
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
        self.models = {}

        reached_max_imf_of_target = False
        for imf_level in data_handler_lstm.train_gen:
            if reached_max_imf_of_target is True:
                break # no need to predict further if target feature doesn't contain greater IMF levels

            if data_handler_lstm.target_max_imf_level == imf_level:
                reached_max_imf_of_target = True
                
            if imf_level in IMFS_TO_PREDICT_USING_LSTM:
                print(f'Training model [{imf_level}]')

                # Prediction model
                model = Sequential()
                current_dataset = data_handler_lstm.train_data[imf_level]
                n_features = current_dataset.shape[1]
                
                current_train_gen = data_handler_lstm.train_gen[imf_level]
                current_val_gen = data_handler_lstm.val_gen[imf_level]

                if imf_level in WIN_SIZE_FOR_IMFS:
                    window_size = WIN_SIZE_FOR_IMFS[imf_level]
                else: 
                    window_size = WIN_SIZE_FOR_IMFS['DEFAULT']

                model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(window_size, n_features)))
                model.add(LSTM(64, activation='tanh', input_shape=(window_size, 128)))
                model.add(Dense(16))
                model.add(LeakyReLU())
                model.add(Dense(4))
                model.add(LeakyReLU())
                model.add(Dense(1)) # 1 target feature only
                model.compile(optimizer=OPTIMIZER, loss=LOSS)

                number_of_epochs = MODEL_EPOCHS[imf_level]

                # fit model
                model.fit(current_train_gen, validation_data = current_val_gen, 
                          # steps_per_epoch=10,
                          epochs=number_of_epochs, verbose=1)

                self.models[imf_level] = model
            else:
                # Spline prediction model
                current_train_gen = data_handler_lstm.train_gen[imf_level]
                model = SplineModel(current_train_gen)
                self.models[imf_level] = model

    def get_forecasts(self):
        models = self.models
        data_handler_lstm = self.data_handler_lstm
        data = data_handler_lstm.data
        results = {}

        # initializing results dictionary
        for feature in FEATURES:
            if feature != FEATURES[-1]:
                continue

            for imf_level in models:
                results[imf_level] = {}
                for col in RESULT_COLS:
                    results[imf_level][col] = []

        for imf_level in models:
            model = models[imf_level]

            print(f'Predicting: [{imf_level}]')

            cur_train_gen = data_handler_lstm.train_gen[imf_level]
            cur_val_gen = data_handler_lstm.val_gen[imf_level]
            cur_test_gen = data_handler_lstm.test_gen[imf_level]

            # predicting train
            day_counter = 0
            for i in tqdm(range(len(cur_train_gen))):
                x, y = cur_train_gen[i] # x is 1, 2, 6 y is (1,1) # 0,1 : 2, 1,2: 3
                yhat = model.predict(x, verbose=0)

                results[imf_level][RESULT_COLS[0]] += [y[0][0]]
                results[imf_level][RESULT_COLS[1]] += [yhat[0][0]]
                results[imf_level][RESULT_COLS[2]] += [day_counter]
                day_counter += 1

            # predicting validation
            for i in tqdm(range(len(cur_val_gen))):
                x, y = cur_val_gen[i]
                yhat = model.predict(x, verbose=0)

                results[imf_level][RESULT_COLS[3]] += [y[0][0]]
                results[imf_level][RESULT_COLS[4]] += [yhat[0][0]]
                results[imf_level][RESULT_COLS[5]] += [day_counter]
                day_counter += 1

            # predicting test
            len_test_gen = len(cur_test_gen)
            for i in tqdm(range(len_test_gen)):
                x, y = cur_test_gen[i]
                yhat = model.predict(x, verbose=0)
                results[imf_level][RESULT_COLS[6]] += [y[0][0]]
                results[imf_level][RESULT_COLS[7]] += [yhat[0][0]]
                results[imf_level][RESULT_COLS[8]] += [day_counter]
                day_counter += 1
                
                if i == len_test_gen - 1: # add one more prediction!
                    x1 = np.zeros(x.shape)
                    cur_win_size = x.shape[1]
                    for j in range(cur_win_size-1):
                        x1[0][j] = x[0][j+1].reshape(1, -1)
                    x1[0][cur_win_size-1] = data_handler_lstm.test_data[imf_level][-1].reshape(1,-1)
                    yhat = model.predict(x1, verbose=0)
                    # print("x1", x1, "YHAT", yhat)
                    results[imf_level][RESULT_COLS[6]] += [np.nan]
                    results[imf_level][RESULT_COLS[7]] += [yhat[0][0]]
                    results[imf_level][RESULT_COLS[8]] += [day_counter]

        concatenated_results = {}
        for imf_level in results: # IMF1,..., Rsd
            # keys of the dictionary become index using orient='index' keys: train_actual, train_pred,...
            df_result = pd.DataFrame.from_dict(results[imf_level], orient='index').T
            df_result = df_result.astype({RESULT_COLS[2]: 'int32'}, copy=False) # change to int from float

            df_train = df_result[[RESULT_COLS[0], RESULT_COLS[1], RESULT_COLS[2]]].set_index(RESULT_COLS[2])
            df_train.index.name = 'x'

            df_val = df_result[[RESULT_COLS[3], RESULT_COLS[4], RESULT_COLS[5]]].set_index(RESULT_COLS[5]).dropna(axis=0)
            df_val.index = df_val.index.astype('int32', copy=False)
            df_val.index.name = 'x'

            df_test = df_result[[RESULT_COLS[6], RESULT_COLS[7], RESULT_COLS[8]]].set_index(RESULT_COLS[8])
            last_day = results[imf_level][RESULT_COLS[8]][-1]
            assert(np.isnan(df_test.loc[last_day][RESULT_COLS[6]]))
            df_test.loc[last_day][RESULT_COLS[6]] = 0 # temporary
            df_test.dropna(inplace=True)
            df_test.loc[last_day][RESULT_COLS[6]] = np.nan # change it back
            df_test.index = df_test.index.astype('int32', copy=False)
            df_test.index.name = 'x'

            # first train rows are not Nan, next validation rows are not Nan, finally test rows are not Nan
            df_concatenated = pd.concat([df_train, df_val, df_test], axis=1)
            print(imf_level, df_concatenated.shape) # loose 6 (Window_size * 3) for IMF1

            concatenated_results[imf_level] = df_concatenated

        add_train_actual_arr = None
        add_val_actual_arr = None
        add_test_actual_arr = None

        add_train_pred_arr = None
        add_val_pred_arr = None
        add_test_pred_arr = None


        for imf_level in concatenated_results:
            ser_imf = concatenated_results[imf_level]
            skip = SKIP_REMAIN_FOR_IMFS[imf_level]

            # adding train actual
            arr_tr = ser_imf[RESULT_COLS[0]].values
            n_train = (~np.isnan(arr_tr)).sum()

            new_arr_tr = arr_tr[skip:n_train]
            if add_train_actual_arr is None:
                add_train_actual_arr = new_arr_tr
                print("r tr", imf_level, len(add_train_actual_arr))
            else:
                add_train_actual_arr = np.add(add_train_actual_arr, new_arr_tr)

            # train pred
            arr_tr = ser_imf[RESULT_COLS[1]].values
            assert(n_train == (~np.isnan(arr_tr)).sum())

            new_arr_tr = arr_tr[skip:n_train]
            if add_train_pred_arr is None:
                add_train_pred_arr = new_arr_tr
                print("tr", imf_level, len(add_train_pred_arr))
            else:
                add_train_pred_arr = np.add(add_train_pred_arr, new_arr_tr)

            # adding val actual
            arr_val = ser_imf[RESULT_COLS[3]].values
            n_val = (~np.isnan(arr_val)).sum()

            new_arr_val = arr_val[skip+n_train:n_val+n_train]
            if add_val_actual_arr is None:
                add_val_actual_arr = new_arr_val
                print("r val", len(add_val_actual_arr))
            else:
                add_val_actual_arr = np.add(add_val_actual_arr, new_arr_val)

            # val pred
            arr_val = ser_imf[RESULT_COLS[4]].values
            assert(n_val == (~np.isnan(arr_val)).sum())

            new_arr_val = arr_val[skip+n_train:n_val+n_train]
            if add_val_pred_arr is None:
                add_val_pred_arr = new_arr_val
                print("val", len(add_val_pred_arr), n_val)
            else:
                add_val_pred = np.add(add_val_pred_arr, new_arr_val)

            # adding test actual
            arr_tst = ser_imf[RESULT_COLS[6]].values
            n_tst = (~np.isnan(arr_tst)).sum()

            new_arr_tst = arr_tst[skip+n_train+n_val:n_tst+n_train+n_val+1] # including the last nan
            if add_test_actual_arr is None:
                add_test_actual_arr = new_arr_tst
                print("r tst", len(add_test_actual_arr), n_tst+1)
            else:
                add_test_actual_arr = np.add(add_test_actual_arr, new_arr_tst)

            # test pred
            arr_tst = ser_imf[RESULT_COLS[7]].values
            assert(n_tst+1 == (~np.isnan(arr_tst)).sum())
            
            new_arr_tst = arr_tst[skip+n_train+n_val:n_tst+n_train+n_val+1]
            if add_test_pred_arr is None:
                add_test_pred_arr = new_arr_tst
                print("tst", len(add_test_pred_arr))
            else:
                add_test_pred_arr = np.add(add_test_pred_arr, new_arr_tst)
                print("tst", imf_level, len(add_test_pred_arr), len(new_arr_tst))

        # inverse transform
        scaler = data_handler_lstm.scalerTgt
        final_prediction_results = {
            RESULT_COLS[0]: scaler.inverse_transform(add_train_actual_arr.reshape(-1,1)).reshape(-1),
            RESULT_COLS[1]: scaler.inverse_transform(add_train_pred_arr.reshape(-1,1)).reshape(-1),
            RESULT_COLS[3]: scaler.inverse_transform(add_val_actual_arr.reshape(-1,1)).reshape(-1),
            RESULT_COLS[4]: scaler.inverse_transform(add_val_pred_arr.reshape(-1,1)).reshape(-1),
            RESULT_COLS[6]: scaler.inverse_transform(add_test_actual_arr.reshape(-1,1)).reshape(-1),
            RESULT_COLS[7]: scaler.inverse_transform(add_test_pred_arr.reshape(-1,1)).reshape(-1),
        }

        train_actual = final_prediction_results[RESULT_COLS[0]].reshape(-1,1)
        train_pred = final_prediction_results[RESULT_COLS[1]].reshape(-1,1)

        train_df = pd.DataFrame(np.hstack([train_actual, train_pred]), columns=[RESULT_COLS[0], RESULT_COLS[1]])

        assert(WIN_SIZE_FOR_IMFS['Rsd'] == 6)
        assert(data_handler_lstm.target_max_imf_level == 'Rsd')

        max_win_size = WIN_SIZE_FOR_IMFS['Rsd']
        cut, val_size = data_handler_lstm.get_train_val_size()
        train_dates = data['Date'][max_win_size:cut] # loose 6 rows from the top due to WIN_SIZE_FOR_IMFS['Rsd']
        train_df['Date'] = train_dates.values

        val_actual = final_prediction_results[RESULT_COLS[3]].reshape(-1,1)
        val_pred = final_prediction_results[RESULT_COLS[4]].reshape(-1,1)

        val_df = pd.DataFrame(np.hstack([val_actual, val_pred]), columns=[RESULT_COLS[3], RESULT_COLS[4]])

        val_dates = data['Date'][max_win_size+cut:cut+val_size]
        val_df['Date'] = val_dates.values

        test_actual = final_prediction_results[RESULT_COLS[6]].reshape(-1,1)
        test_pred = final_prediction_results[RESULT_COLS[7]].reshape(-1,1)

        test_df = pd.DataFrame(np.hstack([test_actual, test_pred]), columns=[RESULT_COLS[6], RESULT_COLS[7]])

        test_dates = data['Date'][max_win_size+cut+val_size:]
        test_df['Date'] = test_dates.values

        assert(data_handler_lstm.timeframe == -1)
        data['Back_Shifted'] = data[data_handler_lstm.target].shift(data_handler_lstm.timeframe)
        df_pred = pd.concat([train_df, val_df, test_df], ignore_index=True)
        df_pred.set_index('Date', inplace=True)

        df_actual = pd.DataFrame() # gather the slices for the actual stock data; given the range of dates
        for ti in [train_dates, val_dates, test_dates]:
            tmp_slice = data[data['Date'].isin(ti)]
            df_actual = pd.concat([df_actual,tmp_slice], ignore_index=True)
            
        df_actual.set_index('Date', inplace=True)

        # join predicted and actual
        recompiled = df_pred.join(df_actual)

        # align df for processing   
        for col in [RESULT_COLS[0],RESULT_COLS[3], RESULT_COLS[6]]: 
            recompiled[col] = recompiled.apply(lambda x: np.nan if np.isnan(x[col]) else x['Back_Shifted'], axis=1)

        # change prediction back into original feature space
        for col in [RESULT_COLS[1],RESULT_COLS[4],RESULT_COLS[7]]:
            recompiled[col] = recompiled[col] * recompiled['Close']

        return recompiled