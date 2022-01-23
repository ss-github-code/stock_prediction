import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class DataHandler:
    def __init__(self, data, target, timeframe, log_return, test_size):
        self.data = data
        self.target = target
        self.timeframe = timeframe
        self.log_return = log_return

        self.data.set_index(['Date'], inplace=True)
        self.data['target'] = self.normalize_target()

        features = [col for col in self.data.columns if col != target]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.data[features], self.data['target'], 
            test_size=test_size, shuffle=False
        )

        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_val[features], self.y_val, 
            test_size=0.5, shuffle=False
        )

    def normalize_target(self):
        target = self.target
        data = self.data
        timeframe = self.timeframe

        if target != 'High':
            assert(0)
        else:
            if self.log_return:
                return np.log(data[target].shift(timeframe) / data['Close'])
            # else: # simple return
            return data[target].shift(timeframe) / data['Close']

    def unnormalize_target(self, data):
        if self.target != 'High':
            assert(0)
        else:
            if self.log_return:
                return np.exp(data['target']) * data['Close']
            return data['target'] * data['Close']

    def calculate_results(self, forecast, actual):
        results = {}

        results['mse'] = mean_squared_error(actual, forecast)
        results['mae'] = mean_absolute_error(actual, forecast)
        results['rmse'] = mean_squared_error(actual, forecast, squared=False)
        results['mape'] = np.mean(np.abs(forecast - actual)/np.abs(forecast))

        return results

    def process_forecasts(self, forecasts):
        val_forecasts = forecasts[:len(self.y_val)]
        test_forecasts = forecasts[len(self.y_val):]

        y_val_df = self.y_val.to_frame()
        y_val_df = y_val_df.join(self.X_val[['Close']]) # columns: target, Close

        val_forecasts.index = y_val_df.index
        val_forecasts_df = val_forecasts.to_frame()

        val_forecasts_df = val_forecasts_df.join(self.X_val[['Close']]) # columns: predicted_mean, Close
        val_forecasts_df.rename(columns={'predicted_mean':'target'}, inplace=True)

        forecast = self.unnormalize_target(val_forecasts_df)
        actual = self.unnormalize_target(y_val_df)
        val_results = self.calculate_results(forecast, actual)

        y_test_df = self.y_test.to_frame()
        y_test_df = y_test_df.join(self.X_test[['Close']]) # columns: target, Close

        test_forecasts.index = y_test_df.index
        test_forecasts_df = test_forecasts.to_frame()
        test_forecasts_df = test_forecasts_df.join(self.X_test[['Close']]) # columns: predicted_mean, Close
        test_forecasts_df.rename(columns={'predicted_mean':'target'}, inplace=True)

        forecast = self.unnormalize_target(test_forecasts_df)
        actual = self.unnormalize_target(y_test_df.iloc[:-1])
        test_results = self.calculate_results(forecast[:-1], actual)

        return forecast, val_results, test_results