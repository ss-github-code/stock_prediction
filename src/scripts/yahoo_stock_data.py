import yfinance as yf
from pandas_datareader import data as pdr
from datetime import date

class YahooStockData:
    def __init__(self, ticker):
        self.ticker = ticker # MSFT

    def get_data(self, start_date, end_date=None):
        yf.pdr_override()
        if end_date is None:
            data = pdr.get_data_yahoo(self.ticker, start = start_date)
        else:
            data = pdr.get_data_yahoo(self.ticker, start = start_date, end = end_date)
        return data