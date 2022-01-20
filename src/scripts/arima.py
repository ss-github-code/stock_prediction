import warnings
warnings.filterwarnings("ignore")

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

class AlgoARIMA:
    def __init__(self, ts, p, d, q):
        assert(d == 0)
        model_autoARIMA = auto_arima(ts, 
                                     start_p=0, max_p=20,
                                     start_q=0, max_q=60,
                                     test='adf', # adf to find optimal 'd'
                                     m=1, # period for seasonal differencing
                                     d=1, #d=None, # let model determine 'd' #d = 1 is first order log difference (log return equivalent)
                                     seasonal=True,
                                     start_P=0,
                                     D=1,
                                     trace=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     )
        # Get optimal ARIMA model params
        best_order = model_autoARIMA.get_params().get('order')
        best_seasonal_order = model_autoARIMA.get_params().get('seasonal_order')

        # Re-run model with optimal params
        # model = ARIMA(ts, order=best_order)
        # Recommended optimization model: SARIMAX
        self.model = SARIMAX(ts,
                             order=best_order,
                             seasonal_order=best_seasonal_order)
        self.model_fitted = self.model.fit()


    def get_forecasts(self, num_forecasts):
        forecasts = self.model_fitted.forecast(steps=num_forecasts)
        return forecasts