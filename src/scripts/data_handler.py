import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

class DataHandler:
    def __init__(self, data, target, timeframe, log_return, test_size):
        self.data = data
        self.target = target
        self.timeframe = timeframe
        self.log_return = log_return

        self.data.set_index(['Date'], inplace=True)
        self.data['target'] = self.normalize_target()
        self.adf_test()
        
        self.features = [col for col in self.data.columns if col != 'target']
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.data[self.features], self.data['target'], 
            test_size=test_size, shuffle=False
        )

        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_val[self.features], self.y_val, 
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

    def adf_test(self, print_results = False):
        # Stationarity test
        ad_fuller_result = adfuller(self.data['target'][:-1], autolag='AIC') # [:-1] to ignore Nan
        output = pd.Series(ad_fuller_result[0:4], index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
        if print_results:
            print('Output of Augmented Dickey Fuller Test')
            print(output)
            for key,values in ad_fuller_result[4].items():
                output['critical value (%s)'%key] = values

        assert(output['p-value'] < 0.05)
    
    def pacf_acf_plots(self):
        # Check appropriateness of AR via partial autocorrelation graph
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        plot_pacf(self.data['target'][:-1], ax, lags=40)
        ax.set_xlabel(r"Lag")
        ax.set_ylabel(r"Correlation")
        del fig, ax #, plot_pacf

        # Check appropriateness of MA via autocorrelation graph
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        plot_acf(self.data['target'][:-1], ax, lags=40)
        ax.set_xlabel(r"Lag")
        ax.set_ylabel(r"Correlation")
        del fig, ax

    def calculate_results(self, forecast, actual):
        results = {}

        results['mse'] = mean_squared_error(actual, forecast)
        results['mae'] = mean_absolute_error(actual, forecast)
        results['rmse'] = mean_squared_error(actual, forecast, squared=False)
        results['mape'] = np.mean(np.abs(forecast - actual)/np.abs(forecast))

        return results

    def process_forecasts(self, forecasts, plot=True):
        val_forecasts = forecasts[:len(self.y_val)]
        test_forecasts = forecasts[len(self.y_val):]

        y_val_df = self.y_val.to_frame()
        y_val_df = y_val_df.join(self.X_val[['Close']]) # columns: target, Close

        val_forecasts.index = y_val_df.index
        val_forecasts_df = val_forecasts.to_frame()

        val_forecasts_df = val_forecasts_df.join(self.X_val[['Close']]) # columns: predicted_mean, Close
        val_forecasts_df.rename(columns={'predicted_mean':'target'}, inplace=True)

        val_forecast_ser = self.unnormalize_target(val_forecasts_df)
        val_actual_ser = self.unnormalize_target(y_val_df)
        val_results = self.calculate_results(val_forecast_ser, val_actual_ser)

        y_test_df = self.y_test.to_frame()
        y_test_df = y_test_df.join(self.X_test[['Close']]) # columns: target, Close

        test_forecasts.index = y_test_df.index
        test_forecasts_df = test_forecasts.to_frame()
        test_forecasts_df = test_forecasts_df.join(self.X_test[['Close']]) # columns: predicted_mean, Close
        test_forecasts_df.rename(columns={'predicted_mean':'target'}, inplace=True)

        test_forecast_ser = self.unnormalize_target(test_forecasts_df)
        test_actual_ser = self.unnormalize_target(y_test_df.iloc[:-1])
        test_results = self.calculate_results(test_forecast_ser[:-1], test_actual_ser)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(18,6))
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            train = self.X_train[['High']][1:]
            plt.plot(train.index, train, color='blue', label='Train', alpha=0.5)
            plt.plot(val_forecast_ser.index, val_forecast_ser, color='black', alpha=0.8, label='Validation predict')
            plt.plot(val_actual_ser.index, val_actual_ser, color='yellow', alpha=0.5, label='Validation actual')
            plt.plot(test_forecast_ser.index, test_forecast_ser, color='red', alpha=0.8, label = 'Test predict')
            plt.plot(test_forecast_ser.index[:-1], test_actual_ser, color='yellow', alpha=0.5, label='Test actual')
            plt.legend()
            plt.show()

        return test_forecast_ser, val_results, test_results