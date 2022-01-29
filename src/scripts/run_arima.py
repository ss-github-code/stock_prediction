import numpy as np
import pandas as pd

import sys, getopt

from yahoo_stock_data import YahooStockData
from data_handler import DataHandler
from arima import AlgoARIMA

START_DATE = '2000-01-01'
TARGET     = 'High'
TIMEFRAME  = -1
LOG_RETURN = True
TEST_SIZE  = 0.2

def run_arima(ticker):
    si_from_yahoo = YahooStockData(ticker)
    si_data = si_from_yahoo.get_data(START_DATE)
    si_data.reset_index(inplace=True)

    data_handler = DataHandler(si_data, TARGET, TIMEFRAME, LOG_RETURN, TEST_SIZE)
    algo_arima = AlgoARIMA(data_handler.y_train)
    forecasts = algo_arima.get_forecasts(len(data_handler.y_val) + len(data_handler.y_test))

    forecast, _, test_results = data_handler.process_forecasts(forecasts, plot_title=f'ARIMA {ticker}')
    print()
    print('************************************')
    print(pd.DataFrame(test_results, index=['Test results'])) # must pass an index (for all scalar values) or change the columns to be a list
    print()
    print('Predicted value:')
    print(forecast[-1])
    return

if __name__ == '__main__':
    run_arima(sys.argv[1])
