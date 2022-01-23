import numpy as np
import pandas as pd

import sys, getopt

from yahoo_stock_data import YahooStockData
from data_handler_lstm import DataHandler_LSTM
from lstm import AlgoLSTM

START_DATE = '2000-01-01'
TARGET     = 'High'
TIMEFRAME  = -1
LOG_RETURN = False
TEST_SIZE  = 0.2
WINDOW_SIZE = 2
NUM_EPOCHS = 2

def run_lstm(ticker):
    si_from_yahoo = YahooStockData(ticker)
    si_data = si_from_yahoo.get_data(START_DATE)
    si_data.reset_index(inplace=True)
    # si_data.to_csv(ticker + '.csv', index=False)
    # si_data = pd.read_csv(ticker + '.csv')
    #print(si_data.columns)
    #print(si_data.shape)
    data_handler = DataHandler_LSTM(si_data, TARGET, TIMEFRAME, LOG_RETURN, TEST_SIZE, WINDOW_SIZE)
    algo_lstm = AlgoLSTM(data_handler, NUM_EPOCHS)
    df_concatenated = algo_lstm.get_forecasts()

    df_forecast, accuracy = data_handler.process_forecasts(df_concatenated)
    print(accuracy)
    print(df_forecast.iloc[-1][TARGET])
    return

if __name__ == '__main__':
    run_lstm(sys.argv[1])