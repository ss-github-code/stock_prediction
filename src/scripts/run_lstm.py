import numpy as np
import pandas as pd

import sys, getopt

from yahoo_stock_data import YahooStockData
from data_handler_lstm import DataHandler_LSTM
from lstm import AlgoLSTM

START_DATE = '2000-01-01'
TARGET     = 'High'
TIMEFRAME  = -1
LOG_RETURN = True
TEST_SIZE  = 0.2
WINDOW_SIZE = 2
NUM_EPOCHS = 4

def run_lstm(ticker):
    si_from_yahoo = YahooStockData(ticker)
    si_data = si_from_yahoo.get_data(START_DATE)
    si_data.reset_index(inplace=True)
    si_data = si_data.astype({'Volume': 'float64'}, copy=False) # required for EMD
    
    # si_data.to_csv(ticker + '.csv', index=False)
    # si_data = pd.read_csv(ticker + '.csv')

    data_handler = DataHandler_LSTM(si_data, TARGET, TIMEFRAME, LOG_RETURN, TEST_SIZE, WINDOW_SIZE)
    algo_lstm = AlgoLSTM(data_handler, NUM_EPOCHS) # initialize the Tensorflow LSTM model
    df_concatenated = algo_lstm.get_forecasts()

    df_forecast, accuracy = data_handler.process_forecasts(df_concatenated, plot_title=f'LSTM {ticker}')
    print('************************************')
    print(pd.DataFrame(accuracy))
    print()
    print('Predicted value:')
    print(df_forecast.iloc[-1]['test_pred'])
    return

if __name__ == '__main__':
    run_lstm(sys.argv[1])