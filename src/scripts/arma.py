import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA

class AlgoARMA:
    def __init__(self, ts, p, d, q):
        assert(d == 0)
        self.arma_model = ARIMA(ts, order=(p,d,q))
        self.arma_fitted = self.arma_model.fit()

    def get_forecasts(self, num_forecasts):
        forecasts = self.arma_fitted.forecast(steps=num_forecasts)
        return forecasts