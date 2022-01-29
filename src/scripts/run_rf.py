import numpy as np
import pandas as pd
import datetime

import sys
sys.path.append('../scripts')

from yahoo_stock_data import YahooStockData
from alphavantage import get_quarterly_reports
from data_handler import DataHandler
from ta_features import add_ta_features
from feature_selection import FeatureSelector
from random_forest import AlgoRandomForest

# constant
TIMEFRAME = -1
TARGET = 'High'
LOG_RETURN = True
TEST_SIZE = 0.2
MAX_NUM_FEATURES = 50

'''
Merge stock data with fundamentals data based on the quarter start and end dates.
Note that we do need to take care of the fact that not all quarter start and end dates will be business days.
'''
def merge_fundamentals(si_data, fun_data): # stock data is reported for every business day, fundamentals are quarterly
    def add_fundamentals(row, params, nrows):
        if row['Date'] >= params[1]: # params[1] is the start of the next quarter
            params[0] = params[1]
            params[2] += 1
            if params[2] < nrows-1:
                params[1] = fun_data.iloc[params[2]+1]['fiscalDateEnding']
            else:
                params[1] = pd.to_datetime("today") + datetime.timedelta(days=1)
            #print(cur_idx, fun_data.iloc[cur_idx]['fiscalDateEnding'])
        for col in fun_data.columns:
            row[col] = fun_data.iloc[params[2]][col]
        return row

    nrows = fun_data.shape[0]
    cur_quarter = fun_data.iloc[0]['fiscalDateEnding'] # 2016-06-30
    nxt_quarter = fun_data.iloc[1]['fiscalDateEnding'] # 2016-09-30
    cur_idx = 0
    params = [cur_quarter, nxt_quarter, cur_idx]
    si_data = si_data.apply(add_fundamentals, axis=1, args = (params, nrows))
    return si_data


def run_rf(ticker, alpha_key):
    fund_df = get_quarterly_reports(ticker, alpha_key) # first get the quarterly reports using Alpha Vantage (only for the last 5 years)
    start_date = fund_df.iloc[0]['fiscalDateEnding'].strftime('%Y-%m-%d') # note that the fund_df is already sorted by date

    si_from_yahoo = YahooStockData(ticker)
    si_data = si_from_yahoo.get_data(start_date) # get the stock data starting from the start date of the quarterly reports
    si_data.reset_index(inplace=True)

    merged_data = merge_fundamentals(si_data, fund_df) # merge stock data with fundamentals data
    merged_data.drop(columns=['fiscalDateEnding'], inplace=True)

    # Add technical analysis features using 'High', 'Low', 'Close', 'Volume' for the stock.
    # The technical analysis features include momentum, volatility, and volume indicators.
    data_w_features = add_ta_features(merged_data)
    print(f'After technical analysis there are {data_w_features.shape[1]} features')

    data_handler = DataHandler(data_w_features, TARGET, TIMEFRAME, LOG_RETURN, TEST_SIZE)

    fselector = FeatureSelector(data_handler, MAX_NUM_FEATURES, show_plot=False) # select up to 50 most important features
    fselector.important_features.extend(['target', 'High', 'Close'])

    data_w_reduced_features = data_w_features[fselector.important_features]
    print(f'After feature selection, there are {data_w_reduced_features.shape[1]} features')

    data_w_reduced_features.reset_index(inplace=True) # 'Date' needs to back as a column
    data_handler = DataHandler(data_w_reduced_features, TARGET, TIMEFRAME, LOG_RETURN, TEST_SIZE)

    algoRF = AlgoRandomForest(data_handler)
    forecasts = algoRF.get_forecasts()
    forecast, _, test_results = data_handler.process_forecasts(forecasts, plot_title=f'Random Forest {ticker}')

    print()
    print('************************************')
    print(pd.DataFrame(test_results, index=['Test results'])) # must pass an index (for all scalar values) or change the columns to be a list
    print()
    print('Predicted value:')
    print(forecast[-1])    

if __name__ == '__main__':
    run_rf(sys.argv[1], sys.argv[2])