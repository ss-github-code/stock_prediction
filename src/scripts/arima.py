import warnings
warnings.filterwarnings("ignore")

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

'''
Autoregression as a modelling technique that leverages observations at adjacent time steps. 
This means that previous time steps inform the dependent variable (future price prediction). 
The target variable used was ln(next_day_high/today_close), so we are effectively predicting
the log change between today’s close and tomorrow’s high, which through the data handler will 
be converted into tomorrow’s high prediction.

AR(p) makes predictions using previous values of the dependent variable. 
MA(q) makes predictions using the series mean and previous errors.
'''
class AlgoARIMA:
    def __init__(self, ts):
        # Automatically discover the optimal order for an ARIMA model
        # The auto_arima function itself operates a bit like a grid search, in that it tries 
        # various sets of p and q (also P and Q for seasonal models) parameters, selecting the model that minimizes the AIC 
        model_autoARIMA = auto_arima(ts, 
                                     start_p=0, max_p=20,
                                     start_q=0, max_q=60,
                                     test='adf', # adf to find optimal 'd'
                                     m=1, # period for seasonal differencing
                                     d=1, #d=None, # let model determine 'd' #d = 1 is first order log difference (log return equivalent)
                                     seasonal=True,
                                     start_P=0, # the starting value of P, the order of the auto-regressive portion of the seasonal model.
                                     D=1, # order of seasonal differencing
                                     trace=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     )
        # Get optimal ARIMA model params
        best_order = model_autoARIMA.get_params().get('order')
        best_seasonal_order = model_autoARIMA.get_params().get('seasonal_order')

        print("Best ARIMA order", best_order)
        print("Best seasonal order", best_seasonal_order)

        # Re-run model with optimal params
        # model = ARIMA(ts, order=best_order)
        # Recommended optimization model: SARIMAX
        # SARIMAX(Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors) is an updated version of the ARIMA model.
        self.model = SARIMAX(ts,
                             order=best_order,
                             seasonal_order=best_seasonal_order)
        self.model_fitted = self.model.fit()


    def get_forecasts(self, num_forecasts):
        forecasts = self.model_fitted.forecast(steps=num_forecasts)
        return forecasts