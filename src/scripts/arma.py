import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA

'''
Autoregression as a modelling technique that leverages observations at adjacent time steps. 
This means that previous time steps inform the dependent variable (future price prediction). 
The target variable used was ln(next_day_high/today_close), so we are effectively predicting
the log change between today’s close and tomorrow’s high, which through the data handler will 
be converted into tomorrow’s high prediction.

AR(p) makes predictions using previous values of the dependent variable. 
MA(q) makes predictions using the series mean and previous errors.
'''
class AlgoARMA:
    # p is the order of the autoregressive polynomial,
    # q is the order of the moving average polynomial.
    def __init__(self, ts, p, q):
        assert(q == 1)
        self.arma_model = ARIMA(ts, order=(p,0,q))
        self.arma_fitted = self.arma_model.fit()

    def get_forecasts(self, num_forecasts):
        forecasts = self.arma_fitted.forecast(steps=num_forecasts)
        return forecasts