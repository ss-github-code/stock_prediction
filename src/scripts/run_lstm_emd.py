import numpy as np
import pandas as pd

import sys

from yahoo_stock_data import YahooStockData
from data_handler_lstm import DataHandler_LSTM
from lstm_emd import AlgoLSTM_EMD

START_DATE = '2000-01-01'
TARGET     = 'High'
TIMEFRAME  = -1
LOG_RETURN = False
TEST_SIZE  = 0.2
WINDOW_SIZE = 2
NUM_EPOCHS = 2

def run_lstm(ticker, show_plot=True):
    si_from_yahoo = YahooStockData(ticker)
    si_data = si_from_yahoo.get_data(START_DATE)
    si_data.reset_index(inplace=True)
    # si_data.to_csv(ticker + '.csv', index=False)
    # si_data = pd.read_csv(ticker + '.csv')
    si_data = si_data.astype({'Volume': 'float64'}, copy=False) # required for EMD

    data_handler = DataHandler_LSTM(si_data, TARGET, TIMEFRAME, LOG_RETURN, TEST_SIZE, WINDOW_SIZE, use_EMD = True)
    algo_lstm = AlgoLSTM_EMD(data_handler)
    df_recompiled = algo_lstm.get_forecasts()

    accuracy = data_handler.calculate_results(df_recompiled, plot=show_plot, plot_title=f'LSTM (EMD) {ticker}')
    
    print('************************************')
    print(pd.DataFrame(accuracy))
    print()
    print('Predicted value:')
    print(df_recompiled.iloc[-1]['test_pred'])

    res_dict = {}
    for k, v in accuracy.items():
        res_dict[k] = v['test']
    return df_recompiled.iloc[-1]['test_pred'], res_dict

if __name__ == '__main__':
    run_lstm(sys.argv[1])