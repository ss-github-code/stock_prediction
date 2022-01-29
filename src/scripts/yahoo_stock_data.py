import yfinance as yf
from pandas_datareader import data as pdr
from datetime import date

'''
Yahoo Finance provides real time low latency API for stock market quotes, crypto currencies, and currency exchange.
We need to provide a start date, the end date is optional (it will download all the data available till today).
It takes a stock market ticker symbol (e.g. MSFT for Microsoft) and returns stock quote for the given time period.
The data consists of Open, Close, High, Low, Volume for each business day since the start date.
'''
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