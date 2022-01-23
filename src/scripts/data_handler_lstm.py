import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from keras.preprocessing.sequence import TimeseriesGenerator

FEATURES = ['Open','High','Low','Close','Volume', 'Target']
RESULT_COLS = ['train_actual', 'train_pred', 'train_x_axis', \
               'val_actual', 'val_pred', 'val_x_axis', \
               'test_actual', 'test_pred', 'test_x_axis']

class ManyToOneTimeSeriesGenerator(TimeseriesGenerator):
  def __getitem__(self, idx):
    x, y = super().__getitem__(idx)
    last_element_index = y.shape[1]-1 # y.shape (1,1)
    return x, y[:,last_element_index].reshape(1,-1)

class DataHandler_LSTM:
    def __init__(self, data, target, timeframe, log_return, test_size, window_size):
        assert(target == 'High')
        assert(timeframe == -1)
    
        self.data = data
        self.target = target
        self.timeframe = timeframe
        self.log_return = log_return
        self.test_size = test_size
        self.window_size = window_size

        # self.data.set_index(['Date'], inplace=True)
        self.data[FEATURES[-1]] = self.normalize_target() # last column is the target
        self.build_generators()

    def get_train_val_size(self):
        test_size = self.test_size
        train_size = 1 - test_size # ratio of training samples to total data
        cut = round(train_size*self.data.shape[0])
        val_size = round(test_size*self.data.shape[0]/2)
        return cut, val_size

    def build_generators(self):
        self.scaler = {}
        train_dict = {}
        val_dict = {}
        test_dict = {}

        cut, val_size = self.get_train_val_size()
        for feature in FEATURES:
            train_ser = self.data[feature][:cut].values.reshape(-1,1)

            scaler = MinMaxScaler()
            scaler.fit(train_ser)
            scaler.transform(self.data[feature].values.reshape(-1,1))

            self.scaler[feature] = scaler
            train_dict[feature] = train_ser.squeeze()
        
            val_dict[feature] = self.data[feature][cut:cut+val_size].values
            test_dict[feature] = self.data[feature][cut+val_size:].values

        print("# Training samples:", cut, " # val samples:", val_size, 
              " # test samples:", self.data.shape[0] - cut - val_size)

        #df_index = self.data.index
        train_df = pd.DataFrame(train_dict, columns=FEATURES)
        val_df = pd.DataFrame(val_dict, columns=FEATURES)
        test_df = pd.DataFrame(test_dict, columns=FEATURES)
        # train_df = pd.DataFrame(train_dict, index=df_index[:cut], columns=FEATURES)
        # val_df = pd.DataFrame(val_dict, index=df_index[cut:cut+val_size], columns=FEATURES)
        # test_df = pd.DataFrame(test_dict, index=df_index[cut+val_size:], columns=FEATURES)
        #print(train_df.shape, val_df.shape, test_df.shape)

        self.train_gen = ManyToOneTimeSeriesGenerator(train_df.values,
                                                      train_df.values,
                                                      length = self.window_size,
                                                      batch_size = 1)
        self.val_gen = ManyToOneTimeSeriesGenerator(val_df.values,
                                                    val_df.values,
                                                    length = self.window_size,
                                                    batch_size = 1)
        self.test_gen = ManyToOneTimeSeriesGenerator(test_df.values,
                                                     test_df.values,
                                                     length = self.window_size,
                                                     batch_size = 1)
        return

    def normalize_target(self):
        target = self.target
        data = self.data
        timeframe = self.timeframe

        if target != 'High':
            assert(0)
        else:
            if self.log_return:
                return np.log(data[target].shift(timeframe) / data['Close'])
            # else: # simple return
            return data[target].shift(timeframe) / data['Close']
    '''
    def unnormalize_target(self, data):
        if self.target != 'High':
            assert(0)
        else:
            if self.log_return:
                return np.exp(data['target']) * data['Close']
            return data['target'] * data['Close']
    '''
    def calculate_results(self, df_final_results):
        
        accuracies_detailed = {}

        y_train = df_final_results[RESULT_COLS[0]][~np.isnan(df_final_results[RESULT_COLS[0]])]
        yhat_train = df_final_results[RESULT_COLS[1]][~np.isnan(df_final_results[RESULT_COLS[1]])]

        y_val = df_final_results[RESULT_COLS[3]][~np.isnan(df_final_results[RESULT_COLS[3]])]
        yhat_val = df_final_results[RESULT_COLS[4]][~np.isnan(df_final_results[RESULT_COLS[4]])]

        # need to shave the end because we don't have next day data
        y_test = df_final_results[RESULT_COLS[6]][~np.isnan(df_final_results[RESULT_COLS[6]])]
        yhat_test = df_final_results[RESULT_COLS[7]][~np.isnan(df_final_results[RESULT_COLS[7]])][:-1]

        accuracies_detailed['mse'] = {
                'train':mean_squared_error(y_train, yhat_train),
                'validation':mean_squared_error(y_val, yhat_val),
                'test':mean_squared_error(y_test, yhat_test),
            }        
        accuracies_detailed['rmse'] = {
                'train':mean_squared_error(y_train, yhat_train, squared=False),
                'validation':mean_squared_error(y_val, yhat_val, squared=False),
                'test':mean_squared_error(y_test, yhat_test, squared=False),
            }
        accuracies_detailed['mae'] = {
                'train':mean_absolute_error(y_train, yhat_train),
                'validation':mean_absolute_error(y_val, yhat_val),
                'test':mean_absolute_error(y_test, yhat_test),
            }
        accuracies_detailed['mape'] = {
                'train':np.mean(np.abs((y_train - yhat_train) / y_train)) * 100,
                'validation':np.mean(np.abs((y_val - yhat_val) / y_val)) * 100,
                'test':np.mean(np.abs((y_test - yhat_test) / y_test)) * 100,
            }
        return accuracies_detailed

    def process_forecasts(self, df_concatenated):
        scaler = self.scaler['Target']
        for col in df_concatenated.columns:
            df_concatenated[col] = scaler.inverse_transform(df_concatenated[col].values.reshape(-1,1))

        data = self.data.copy()
        data['Back_Shifted_Actual'] = data[self.target].shift(self.timeframe)

        win_size = self.window_size
        cut, val_size = self.get_train_val_size()

        train_dates = data['Date'][win_size:cut] # skip the win_size rows for which we do not have a prediction
        val_dates = data['Date'][win_size + cut: cut + val_size] # similarly skip win_size rows from validation and test
        test_dates = data['Date'][win_size + cut + val_size:]

        print(len(train_dates), len(val_dates), len(test_dates))

        df_data_slc = pd.DataFrame()
        for ti in [train_dates, val_dates, test_dates]:
            tmp_slice = data[(data['Date'].isin(ti))]
            df_data_slc = pd.concat([df_data_slc, tmp_slice])

        print(df_data_slc.shape)

        df_concatenated = df_concatenated.set_index(df_data_slc['Date'])
        df_data_slc.set_index('Date', inplace=True)

        # join predicted and real
        df_recompiled = df_concatenated.join(df_data_slc)

        # change prediction back into original feature space
        for col in [RESULT_COLS[1], RESULT_COLS[4], RESULT_COLS[7]]: # train_pred, val_pred, test_pred
            if self.log_return:
                df_recompiled[col] = np.exp(df_recompiled[col]) * df_recompiled['Close']
            else:
                df_recompiled[col] = df_recompiled[col] * df_recompiled['Close']
        
        for col in [RESULT_COLS[0], RESULT_COLS[3], RESULT_COLS[6]]: 
            df_recompiled[col] = df_recompiled.apply(lambda x: np.nan if np.isnan(x[col]) else x['Back_Shifted_Actual'], axis=1)

        return df_recompiled, self.calculate_results(df_recompiled)