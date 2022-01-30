from celery import shared_task
import pandas as pd

from stock_predictor.models_helper.yahoo_stock_data import YahooStockData
from stock_predictor.models_helper.data_handler import DataHandler
from stock_predictor.models_helper.arma import AlgoARMA

START_DATE = '2000-01-01'
TARGET     = 'High'
TIMEFRAME  = -1
LOG_RETURN = True
TEST_SIZE  = 0.2

@shared_task
def run_arma(ticker):
    si_from_yahoo = YahooStockData(ticker)
    si_data = si_from_yahoo.get_data(START_DATE)
    si_data.reset_index(inplace=True)

    data_handler = DataHandler(si_data, TARGET, TIMEFRAME, LOG_RETURN, TEST_SIZE)
    algo_arma = AlgoARMA(data_handler.y_train, 2, 0, 1)
    forecasts = algo_arma.get_forecasts(len(data_handler.y_val) + len(data_handler.y_test))

    forecast, _, test_results = data_handler.process_forecasts(forecasts)
    
    return forecast[-1]