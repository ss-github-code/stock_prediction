from wsgiref import validate
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from PyEMD import CEEMDAN # Empirical Mode Decomposition (EMD). Most popular expansion is Ensemble Empirical Mode Decomposition (EEMD)
from keras.preprocessing.sequence import TimeseriesGenerator

FEATURES = ['Open','High','Low','Close','Volume', 'Target']
RESULT_COLS = ['train_actual', 'train_pred', 'train_x_axis', \
               'val_actual', 'val_pred', 'val_x_axis', \
               'test_actual', 'test_pred', 'test_x_axis']
MAX_IMFS = 6 # components are often called Intrinsic Mode Functions (IMF) to highlight 
             # that they contain an intrinsic property which is a specific oscillation (mode)

WIN_SIZE_FOR_IMFS = {
    'IMF1': 2,
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

class ManyToOneTimeSeriesGenerator(TimeseriesGenerator):
  def __getitem__(self, idx):
    x, y = super().__getitem__(idx)
    last_element_index = y.shape[1]-1 # y.shape (1,1)
    return x, y[:,last_element_index].reshape(1,-1)

class DataHandler_LSTM:
    def __init__(self, data, target, timeframe, log_return, test_size, window_size, use_EMD=False):
        assert(target == 'High')
        assert(timeframe == -1)
        if use_EMD:
            assert(log_return == False) 

        self.data = data
        self.target = target
        self.timeframe = timeframe
        self.log_return = log_return
        self.test_size = test_size
        self.window_size = window_size

        # self.data.set_index(['Date'], inplace=True)
        self.data[FEATURES[-1]] = self.normalize_target() # last column is the target
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

    def decompose(self):
        data = self.data
        ceemdan = CEEMDAN(parallel = True, processes=8)
        data[FEATURES[-1]].fillna(0,inplace=True) # cannot have NaN in CEEMDAN

        print('Decomposing using EMD library')
        decomposed_features_series = {}
        for col in FEATURES: # decompose the 6 time series (Open, High, Low, Close, Volume, Target)
            decomposed_features_series[col] = {}
            try:
                series = data[col].values.reshape(-1,1)
                # decompose
                feature_time_series = np.frombuffer(series)
                feature_time_series_imfs = ceemdan(feature_time_series, max_imf=MAX_IMFS)
                # iterating every IMF 
                for i, imf_series in enumerate(feature_time_series_imfs):
                    if i < len(feature_time_series_imfs)-1: # last one is residual
                        decomposed_features_series[col][f'IMF{i+1}'] = imf_series
                    else:
                        decomposed_features_series[col][f'Rsd'] = imf_series
                print(f'Finished Decomposing {col}: #IMFS: {len(feature_time_series_imfs)}')
            except:
                print(f'ERROR decomposing [{col}]')
                decomposed_features_series[col] = 'ERROR'                
            finally:
                continue
        
        cut, val_size = self.get_train_val_size()
        self.scalerTgt = None
        for col in decomposed_features_series.keys():
            for imf in decomposed_features_series[col]:
                if imf != 'Rsd':
                    continue
                train_ser = decomposed_features_series[col][imf][:cut].reshape(-1,1)
                scaler = MinMaxScaler()
                scaler.fit(train_ser)
                decomposed_features_series[col][imf] = scaler.transform(decomposed_features_series[col][imf].reshape(-1,1)).flatten()
                if col == FEATURES[-1]:
                    self.scalerTgt = scaler # save the scaler for inverse_transform after prediction

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
                _series = _series.reshape((len(_series),1)) # reshaping to get into column format
                series[imf] += [_series] # list of (len(data), 1)
                # print(feature, imf, _series.shape, len(series[ticker][imf]))
            if col == FEATURES[-1]:
                self.target_max_imf_level = imf
                assert(self.target_max_imf_level == 'Rsd')

        # horizontal stack
        full_data = {}
        for imf_level in series:
            assert(len(series[imf_level]) == 6)
            full_data[imf_level] = np.hstack(tuple(series[imf_level]))
            print(imf_level, full_data[imf_level].shape) # (len(data), 6)

        self.train_data = {} # needed while modeling for number of input features (6)
        val_data = {}
        test_data = {}

        for imf_level in full_data:
            # splitting data sets according to rates
            self.train_data[imf_level] = full_data[imf_level][:cut, :]
            val_data[imf_level] = full_data[imf_level][cut:cut+val_size, :]
            test_data[imf_level] = full_data[imf_level][cut+val_size:, :]
        
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
            self.test_gen[imf_level] = ManyToOneTimeSeriesGenerator(test_data[imf_level],
                                                                    test_data[imf_level],
                                                                    length = window_size, batch_size = 1)
        return        

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