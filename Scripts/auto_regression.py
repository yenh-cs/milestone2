"""
Estimate an AR-X model using Conditional Maximum Likelihood (OLS).
"""
from typing import Tuple
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class PartialModel:
    """ Allows for parital initialization of scipy SARIMA family of models"""
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.model(x, *self.args, **self.kwargs)

def partial_ar(lags):
    """ partial autoregressive model """
    model = PartialModel(AutoReg, lags=lags)
    return model


def partial_arma(order: Tuple[int, int]):
    """ partial autoregressive moving average model """
    order = (order[0], 0, order[1])
    model = PartialModel(ARIMA, order=order)
    return model


def partial_arima(order: Tuple[int, int, int]):
    """ partial autoregressive integrated moving average model """
    model = PartialModel(ARIMA, order=order)
    return model


def partial_sarimax(
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int]
):
    """ partial seasonal autoregressive integrated moving average model """
    model = PartialModel(SARIMAX, order=order, seasonal_order=seasonal_order)
    return model
