import numpy as np
import pandas as pd

import sys

from yahoo_stock_data import YahooStockData
from data_handler import DataHandler
from arma import AlgoARMA

START_DATE = '2000-01-01'
TARGET     = 'High'
TIMEFRAME  = -1
LOG_RETURN = True
TEST_SIZE  = 0.2

def run_arma(ticker, show_plot=True):
    si_from_yahoo = YahooStockData(ticker)
    si_data = si_from_yahoo.get_data(START_DATE)
    si_data.reset_index(inplace=True)

    data_handler = DataHandler(si_data, TARGET, TIMEFRAME, LOG_RETURN, TEST_SIZE)
    algo_arma = AlgoARMA(data_handler.y_train, 2, 1) # AR(p) and MA(q)
    forecasts = algo_arma.get_forecasts(len(data_handler.y_val) + len(data_handler.y_test))

    forecast, _, test_results = data_handler.process_forecasts(forecasts, plot=show_plot, plot_title=f'ARMA {ticker}')
    print()
    print('************************************')
    print(pd.DataFrame(test_results, index=['Test results'])) # must pass an index (for all scalar values) or change the columns to be a list
    print()
    print('Predicted value:')
    print(forecast[-1])
    return forecast[-1], test_results

if __name__ == '__main__':
    run_arma(sys.argv[1])