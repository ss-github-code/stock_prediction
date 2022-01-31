# A Sentiment Analysis Approach to Stock Prediction

The stock market and its trends are highly volatile. As the stock market produces a large amountof data every day, it is challenging for an individual to consider the current and past information for predicting the future trends of the stock market. In this project, we explore the use of different
statistical and machine learning models that use the stock market data, stock’s fundamentals data and sentiment analysis of news data as input features. We investigate the ability of six different models to predict the next day’s high price of a stock. We use four accuracy metrics as a measure
to evaluate a model’s performance to predict the next day’s high price of a stock.

# Data source
1. Stock data using Yahoo Finance (YF) API
YF provides a real-time low latency API for stock market quotes, cryptocurrencies, and currency exchange. We used the API to download the data using a stock’s ticker symbol (e.g., Microsoft’s ticker is MSFT). We collected data starting from January 2000 to the present. For each stock symbol, we receive the daily Open, High, Low, Close prices, and Volume of trading for the stock. The data is returned as a Pandas data frame and has
about 5500 rows per stock ticker.
2. News data using New York Times (NYT) API
We used the NYT API to fetch news articles from their business section. We collected the articles starting from January 2000 to the present. We used VADER (Valence Aware Dictionary for sEntiment Reasoning) sentiment analysis to compute the average sentiment across the collected new articles for a day. The data is returned in JSON format and we collect about 160000 articles using the API.
3. Stock’s Fundamentals data using Alpha Vantage (AV) API
AV provides an API to download the last 5 years of quarterly reports for a given stock symbol. The quarterly report covers key financial metrics reported by a public traded company every quarter; e.g., gross profit, total revenue, income, and research and development cost. The data is returned in JSON format and has 20 rows per stock ticker.

# Part A - Supervised Learning
## Motivation
Financial time series analysis and forecasting have had several approaches over time. Currently, the most widely used methods to forecasting can be roughly divided into two groups: statistical techniques and AI/ML approaches.
We used the traditional statistical models - the Autoregressive Moving Average (ARMA) and the Autoregressive Integrated Moving Average (ARIMA) models to forecast the next day’s high price of a stock. These models assume that the input time series has been transformed into a stationary time series.More recent AI/ML models focus on learning the behavior of a time series just from its data, without prior explicit assumptions of stationarity. Signal processing techniques have been proposed to do feature engineering of the stock data. We used Random Forest and LSTM based neural networks to predict the same target variable as the statistical models. We explored the use of technical analysis features (momentum, volatility, and volume) of a stock’s data as input to the Random Forest model. We investigated the use of the signal processing technique called Empirical Mode Decomposition (EMD) that can be used to decompose the multivariate time seriesin a stock’s data (Open, Close, High, Low, Volume) and use the decomposed features as input to an LSTM based neural net.
# Part B - Unsupervised Learning
## Motivation
We wanted to explore the effect of today’s news articles on the next day’s stock market. Initially we wanted to include Twitter data in this section. However, we were not successful in collecting historical data from Twitter. In addition, we did not find a data source for collecting historical news
for a given stock. Instead, we used New York Times API to collect news articles from their business section starting from January 2000 to present. The sentiment analysis of the news articles was used as an input feature to the supervised learning methods for all stock symbols. We wanted to investigate the use of a company’s business fundamentals data issued every quarter on its stock price. We used Alpha Vantage API to collect the quarterly reports for each
stock symbol. In addition, we wanted to use key financial indicators derived from the High, Low, Close, and Volume data for a stock that indicate its momentum, and volatility. We used a technical analysis library to find the key indicators in the stock’s data.
