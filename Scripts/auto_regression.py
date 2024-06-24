"""
Estimate an AR-X model using Conditional Maximum Likelihood (OLS).
"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Scripts.misc import mape, mse, rmse, me
from Scripts.constants import utd
from Scripts.datasets import UTDCityDataset, train_val_test_split

from functools import partial


class PartialModel:
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.model(x, *self.args, **self.kwargs)

def acf_pacf():
    dfs = utd.get_city_dfs('paris', True, False, False)
    df_traffic = dfs.traffic_df

    data = df_traffic.flow.diff().dropna()

    xs = sm.tsa.stattools.pacf(data)
    xs = [(i, np.abs(x)) for i, x in enumerate(xs) if i != 0]
    xs = sorted(xs, key=lambda x: x[1], reverse=True)
    lags = [x[0] for x in xs[:5]]
    return lags

def convolute(series, window: int, stride: int):
    out_series = []
    for i in range(0, len(series) + window, stride):
        out_series.append(series[i: i+window].mean())
    return out_series


def run_model(partial_model, viz=False):
    dset = UTDCityDataset('paris', 100, 50)
    train_dset, val_dset, test_dset = train_val_test_split(dset, 0.8, 0.1, 0.1)

    if viz:
        for i in np.random.randint(0, len(test_dset), 5):
            x, y = dset[i]
            model = partial_model(x)
            fitted_model = model.fit()
            y_preds = fitted_model.predict(len(x), len(x) + 50 - 1)
            mape_ = me(y, y_preds)
            plt.plot(np.arange(len(x)), x, "k")
            plt.plot(np.arange(len(x), len(x) + 50), y, "b--")
            plt.plot(np.arange(len(x), len(x) + 50), y_preds, "r--")
            plt.title(f'{mape_}')
            plt.show()

    else:
        avg_mapes = []
        for _ in trange(300):
            mapes = []
            for i in np.random.randint(0, len(test_dset), 100):
                x, y = dset[i]
                model = partial_model(x)
                fitted_model = model.fit()
                y_preds = fitted_model.predict(len(x), len(x) + 50 - 1)
                mape_ = me(y, y_preds)
                mapes.append(mape_)
                avg_mape = np.mean(mapes)
                avg_mapes.append(avg_mape)

        plt.hist(avg_mapes)
        plt.show()
        print(np.mean(avg_mapes))


def ar(lags):
    model = PartialModel(AutoReg, lags=lags)
    return model


def partial_arma(order: Tuple[int, int]):
    order = (order[0], 0, order[1])
    model = PartialModel(ARIMA, order=order)
    return model


def partial_arima(order: Tuple[int, int, int]):
    model = PartialModel(ARIMA, order=order)
    return model


def partial_sarimax(
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int]
):
    model = PartialModel(SARIMAX, order=order, seasonal_order=seasonal_order)
    return model


if __name__ == "__main__":
    from statsmodels.tsa.seasonal import seasonal_decompose
    import sys
    # lags = acf_pacf()
    # ar = ar(lags)

    # dfs = utd.get_city_dfs('paris', True, False, False)
    # df_traffic = dfs.traffic_df
    # df_traffic = df_traffic.loc[df_traffic.detid == 1949][['day', 'interval', 'flow']]
    # flow = df_traffic.flow
    # flow_diff = flow.diff()
    #
    # train_flow = flow_diff.iloc[:7500]
    # test_flow = flow_diff.iloc[7500:]
    #
    # # arma = partial_sarimax((1, 0, 1), (3, 0, 3, 8))
    # arma = partial_sarimax((1, 0, 1), (1, 0, 1, 24))
    # arma = arma(train_flow)
    # fitted_model = arma.fit()
    # print(fitted_model.summary())
    # predict_steps = 100
    # # y_pred_diffs = fitted_model.predict(start=len(train_flow), end=len(train_flow) + predict_steps, dynamic=True)
    # y_pred_diffs = fitted_model.forecast(100)
    # y_pred = [flow.iloc[len(train_flow) - 1]]
    # for diff in y_pred_diffs:
    #     y_pred.append(y_pred[-1] + diff)
    #
    # y_pred = y_pred[1:]
    # y_pred = y_pred[:100]
    # y_true = flow.iloc[len(train_flow): len(train_flow)+len(y_pred)].tolist()
    # plt.plot(y_pred, 'r--')
    # plt.plot(y_true, 'b--')
    # plt.show()


    # run_model(arma)

    # from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    #
    # dfs = utd.get_city_dfs('paris', True, False, False)
    # df_traffic = dfs.traffic_df
    #
    # data = df_traffic.flow.diff().dropna()[:1000]
    # # data = df_traffic.flow.dropna()[:1000]
    # plot_pacf(data)
    # plt.show()
    #
    # plot_acf(data)
    # plt.show()
    # xs = sm.tsa.stattools.acf(data)
    # xs = [(i, np.abs(x)) for i, x in enumerate(xs) if i != 0]
    # xs = sorted(xs, key=lambda x: x[1], reverse=True)
    # lags = [x[0] for x in xs[:5]]
    # print(lags)


