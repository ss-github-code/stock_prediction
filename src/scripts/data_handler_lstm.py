import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from PyEMD import CEEMDAN # Empirical Mode Decomposition (EMD). Most popular expansion is Ensemble Empirical Mode Decomposition (EEMD)
from keras.preprocessing.sequence import TimeseriesGenerator

RESULT_COLS = ['train_actual', 'train_pred', 'train_x_axis', \
               'val_actual', 'val_pred', 'val_x_axis', \
               'test_actual', 'test_pred', 'test_x_axis']
MAX_IMFS = 6 # components are often called Intrinsic Mode Functions (IMF) to highlight 
             # that they contain an intrinsic property which is a specific oscillation (mode)

WIN_SIZE_FOR_IMFS = {
    'IMF1': 2, # use smaller window for high frequency component
    'IMF2': 2,
    'IMF3': 3,
    'IMF4': 3,
    'IMF5': 4,
    'IMF6': 4,
    'IMF7': 5,
    'IMF8': 5,
    'Rsd' : 6,
    'DEFAULT': 4
}
SKIP_REMAIN_FOR_IMFS = {
    'IMF1': 4, # 6 (Rsd) - 2
    'IMF2': 4,
    'IMF3': 3,
    'IMF4': 3,
    'IMF5': 2,
    'IMF6': 2,
    'IMF7': 1,
    'IMF8': 1,
    'Rsd': 0,
    'DEFAULT': 2
}

'''
This class takes in a sequence of data-points gathered at equal intervals, along with time series parameters such as stride, 
length of history, etc., to produce batches for training/validation.
'''
class ManyToOneTimeSeriesGenerator(TimeseriesGenerator):
  def __getitem__(self, idx):
    x, y = super().__getitem__(idx)
    last_element_index = y.shape[1]-1 # y.shape (1,1)
    return x, y[:,last_element_index].reshape(1,-1) # subclassing it so that we only return the last column in batch_size rows

class DataHandler_LSTM:
    def __init__(self, data, target, timeframe, log_return, test_size, window_size, use_EMD=False, use_sentiment=False):
        assert(target == 'High')
        assert(timeframe == -1)
        if use_EMD:
            assert(log_return == False)
            assert(use_sentiment == False) 

        if use_sentiment:
            self.features = ['Open','High','Low','Close','Volume', 'compound', 'Target']
        else:
            self.features = ['Open','High','Low','Close','Volume', 'Target']
        self.data = data
        self.target = target
        self.timeframe = timeframe
        self.log_return = log_return
        self.test_size = test_size
        self.window_size = window_size

        # self.data.set_index(['Date'], inplace=True)
        self.data[self.features[-1]] = self.normalize_target() # last column is the target
        if use_EMD:
            self.decompose()
        else:
            self.build_generators()

    def get_train_val_size(self):
        test_size = self.test_size
        train_size = 1 - test_size # ratio of training samples to total data
        cut = round(train_size*self.data.shape[0])
        val_size = round(test_size*self.data.shape[0]/2)
        return cut, val_size

    '''
    Neural net models try to focus on learning the behavior of a series from its data, without prior explicit assumptions, 
    such as linearity or stationarity. An ideal approach is to divide the tough task of forecasting the original time series 
    into several subtasks, and each of them forecasts a relatively simpler subsequence. 
    And then the results of all subtasks are accumulated as the final result.
    We will use Empirical Mode Decomposition to decompose each of the 6 (Open, Close, High, Low, Volume, Target) into components.
    PyEMD is a Python implementation of Empirical Mode Decomposition (EMD) and its variations. One of the most popular expansion is 
    Ensemble Empirical Mode Decomposition (EEMD), which utilises an ensemble of noise-assisted executions.
    '''
    def decompose(self):
        data = self.data
        features = self.features
        ceemdan = CEEMDAN(parallel = True, processes=8)
        # data[FEATURES[-1]].fillna(0, inplace=True) # cannot have NaN in CEEMDAN

        cut, val_size = self.get_train_val_size()

        # First scale
        scaled_features_series = {}
        self.scalerTgt = None
        for col in features:
            series = data[col].values.reshape(-1,1)
            if col == features[-1]:
                series = series[:-1] # leave out NaN in the target column (cannot have NaN in CEEMDAN)

            feature_time_series = np.frombuffer(series)
            train_ser = feature_time_series[:cut]
            scaler = MinMaxScaler()
            scaler.fit(train_ser.reshape(-1,1))
            scaled_features_series[col] = scaler.transform(feature_time_series.reshape(-1,1)).flatten()
            if col == features[-1]:
                self.scalerTgt = scaler # save the scaler for inverse_transform after prediction        

        # Then decompose each input feature using the EMD library
        print('Decomposing using EMD library')
        decomposed_features_series = {}
        for col in features: # decompose the 6 time series (Open, High, Low, Close, Volume, Target)
            decomposed_features_series[col] = {}
            try:
                # decompose
                feature_time_series = np.frombuffer(scaled_features_series[col])
                feature_time_series_imfs = ceemdan(feature_time_series, max_imf=MAX_IMFS)
                # iterating every IMF 
                for i, imf_series in enumerate(feature_time_series_imfs):
                    if i < len(feature_time_series_imfs)-1: # last one is residual
                        decomposed_features_series[col][f'IMF{i+1}'] = imf_series
                    else:
                        decomposed_features_series[col][f'Rsd'] = imf_series
                print(f'Finished Decomposing {col}: #IMFS: {len(feature_time_series_imfs)}, {len(imf_series)}')
            except:
                print(f'ERROR decomposing [{col}]')
                decomposed_features_series[col] = 'ERROR'                
            finally:
                continue

        # Coupling together the IMFs of the same level for different features to create exogenous input
        series = {}
        self.target_max_imf_level = None      
        for col in decomposed_features_series.keys(): # Open, High,..., Target
            # 6 features, each one has 7 IMFs
            imfs = pd.DataFrame.from_dict(decomposed_features_series[col])
            # print("Feature", feature , imfs.shape) # len(data), 7 (IMF1, .., IMF6, Rsd)
            for imf in imfs:
                if imf not in series:
                    series[imf] = [] # empty list
                _series = imfs[imf].values
                if col != features[-1]: # other than Target, each series has one more entry
                    _series = _series[:-1]                
                _series = _series.reshape((len(_series),1)) # reshaping to get into column format
                series[imf] += [_series] # list of (len(data)-1, 1)
                # print(feature, imf, _series.shape, len(series[ticker][imf]))
            if col == features[-1]:
                self.target_max_imf_level = imf
                assert(self.target_max_imf_level == 'Rsd')

        # horizontal stack
        full_data = {}
        for imf_level in series:
            assert(len(series[imf_level]) == 6)
            full_data[imf_level] = np.hstack(tuple(series[imf_level]))
            print(imf_level, full_data[imf_level].shape) # (len(data)-1, 6)

        self.train_data = {} # needed while modeling for number of input features (6)
        val_data = {}
        self.test_data = {}

        for imf_level in full_data:
            # splitting data sets according to rates
            self.train_data[imf_level] = full_data[imf_level][:cut, :]
            val_data[imf_level] = full_data[imf_level][cut:cut+val_size, :]
            self.test_data[imf_level] = full_data[imf_level][cut+val_size:, :] # Note that test_data has one more entry
        
        self.train_gen = {}
        self.val_gen = {}
        self.test_gen = {}
        for imf_level in full_data:
            if imf_level in WIN_SIZE_FOR_IMFS:
                window_size = WIN_SIZE_FOR_IMFS[imf_level]
            else: 
                window_size = WIN_SIZE_FOR_IMFS['DEFAULT']
            # windowing
            self.train_gen[imf_level] = ManyToOneTimeSeriesGenerator(self.train_data[imf_level], # data
                                                                     self.train_data[imf_level], # target
                                                                     length = window_size, batch_size = 1) # number of timesteps with sampling__rate=1
            self.val_gen[imf_level] = ManyToOneTimeSeriesGenerator(val_data[imf_level],
                                                                   val_data[imf_level],
                                                                   length = window_size, batch_size = 1) 
            self.test_gen[imf_level] = ManyToOneTimeSeriesGenerator(self.test_data[imf_level],
                                                                    self.test_data[imf_level],
                                                                    length = window_size, batch_size = 1)
        return        

    def build_generators(self):
        features = self.features
        data = self.data
        self.scalers_dict = {}
        train_dict = {}
        val_dict = {}
        test_dict = {}

        # First scale by applying min max scaler
        # This estimator scales and translates each feature individually such that it is in the given range on the 
        # training set, e.g. between zero and one.
        cut, val_size = self.get_train_val_size()
        for feature in features:
            series = data[feature].values.reshape(-1,1)
            feature_time_series = np.frombuffer(series)

            scaler = MinMaxScaler()
            self.scalers_dict[feature] = scaler

            train_ser = feature_time_series[:cut].reshape(-1,1)
            scaler.fit(train_ser)
            scaled_feature_ser = scaler.transform(feature_time_series.reshape(-1,1)).flatten()

            train_dict[feature] = scaled_feature_ser[:cut]
            val_dict[feature] = scaled_feature_ser[cut:cut+val_size]
            test_dict[feature] = scaled_feature_ser[cut+val_size:]

        print("# Training samples:", cut, " # val samples:", val_size, 
              " # test samples:", self.data.shape[0] - cut - val_size)

        train_df = pd.DataFrame(train_dict, columns=features)
        val_df = pd.DataFrame(val_dict, columns=features)
        test_df = pd.DataFrame(test_dict, columns=features)
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
    def calculate_results(self, df_final_results, plot=True, plot_title='Title'):
        
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
        if plot:
            today = datetime.datetime.today().strftime('%Y/%m/%d')
            fig, ax = plt.subplots(1, 1, figsize=(18,6))
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            plt.plot(y_train.index, y_train, color='blue', label='Train', alpha=0.5)
            plt.plot(y_val.index, yhat_val, color='black', alpha=0.8, label='Validation predict')
            plt.plot(y_val.index, y_val, color='yellow', alpha=0.5, label='Validation actual')
            plt.plot(y_test.index, yhat_test, color='red', alpha=0.8, label = 'Test predict')
            plt.plot(y_test.index, y_test, color='yellow', alpha=0.5, label='Test actual')
            plt.title(f'{plot_title} {today}')
            plt.legend()
            plt.show()

        return accuracies_detailed

    def process_forecasts(self, df_concatenated, plot=True, plot_title='Title'):
        # Apply scaler inverse transform to the predicted target columns
        scaler = self.scalers_dict['Target']
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

        return df_recompiled, self.calculate_results(df_recompiled, plot=plot, plot_title=plot_title)