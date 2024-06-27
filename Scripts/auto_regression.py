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
from pmdarima.arima import auto_arima

from Scripts.misc import mpe
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
            mape_ = mpe(y, y_preds)
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


def run_auto_arima():
    dfs = utd.get_city_dfs("paris")
    traffic_df = dfs.traffic_df

    detids = traffic_df.detid.unique()

    xs = []
    ys = []

    pred_len = 100
    # results
    # l = [[-0.22522390678658627, -1.4301656799480724, -0.10705467754623049], [-0.1565106551686227, -0.5744326663220725, -0.2145666535343044], [-0.15135927581272118, -0.5093898272465825, -0.19212728359763376]]


    params = [
        {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 24)},
        {'order': (1, 0, 1), 'seasonal_order': (1, 0, 1, 24)},
        {'order': (2, 0, 2), 'seasonal_order': (1, 1, 1, 24)}
    ]

    detids = np.random.choice(detids, 3, replace=True)
    pred_size = 100
    metrics = [[] for x in params]


    for i, param in enumerate(params):
        for detid in detids:
            flow = traffic_df.loc[traffic_df.detid == detid]["flow"].values
            x = flow[:-pred_size]
            y_true = flow[-pred_size:]
            sarima = SARIMAX(x, **param)
            fitted_model = sarima.fit()
            y_pred = fitted_model.forecast(pred_size)
            metrics[i].append(mpe(y_true, y_pred))

    print(metrics)


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

def plot_sarimax_lstm():
    def lstm_predict(model, X):
        mx = X.max()
        X = torch.tensor(X[None, :, None], dtype=torch.float32) / mx
        y_pred = model(X).detach().cpu().numpy().squeeze()
        y_pred = y_pred * mx
        return y_pred

    def knn_predict(knn, X):
        X = X[None, -200:]
        y = knn.predict(X)
        return y.squeeze()

    from statsmodels.tsa.seasonal import seasonal_decompose
    import pickle
    import torch
    import matplotlib.patches as mpatches
    from Scripts.model import LSTM
    import os
    import sys
    # lags = acf_pacf()
    # ar = ar(lags)

    model_weights_p = "./../Models/NoShuffle/1719332148.pt"
    model_state_d = torch.load(model_weights_p, map_location=torch.device('cpu'))
    lstm = LSTM(50, 100, num_layers=3)
    lstm.load_state_dict(model_state_d)

    knn_p = "./../Models/knn.pkl"
    with open(knn_p, 'rb') as f:
        knn = pickle.load(f)

    pred_size = 100
    n = 10

    dfs = utd.get_city_dfs('paris', True, False, False)
    df_traffic = dfs.traffic_df
    detids = df_traffic.detid.unique().tolist()
    # df_traffic = df_traffic.loc[df_traffic.detid == 1949][['day', 'interval', 'flow']]
    # flow = df_traffic.flow
    # flow_diff = flow.diff()
    save_dir = r'/Users/joshfisher/PycharmProjects/Milestone2/Data/Visualizations/sarima_knn_lstm'

    for detid in tqdm(np.random.choice(detids, replace=False, size=n), total=n):
        tmp_df = df_traffic.loc[df_traffic.detid == detid]
        flow = tmp_df.flow
        train_flow = flow.iloc[:-pred_size]
        test_flow = flow.iloc[-pred_size:]

        sarima = SARIMAX(train_flow, order=(2, 0, 2), seasonal_order=(1, 1, 1, 24))
        fitted_model = sarima.fit()

        y_pred = fitted_model.forecast(pred_size)
        lstm_preds = lstm_predict(lstm, flow.iloc[:-pred_size].values)
        knn_preds = knn_predict(knn, flow.iloc[:-pred_size].values)

        x = flow.iloc[-2 * pred_size: -pred_size].values
        y_true = test_flow.values

        f, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(pred_size), x, 'k')
        ax.plot(np.arange(pred_size, 2*pred_size, 1), y_true, 'k--')
        ax.plot(np.arange(pred_size, 2*pred_size, 1), y_pred, 'b--')
        ax.plot(np.arange(pred_size, 2*pred_size, 1), lstm_preds, 'g--')
        ax.plot(np.arange(pred_size, 2*pred_size, 1), knn_preds, 'r--')

        red_patch = mpatches.Patch(color='red', label="KNN")
        blue_patch = mpatches.Patch(color='blue', label='SARIMA')
        green_patch = mpatches.Patch(color='green', label="LSTM")
        ax.legend(handles=[red_patch, green_patch, blue_patch])
        # plt.title(f'lstm: {mpe(y_true, lstm_preds): 0.2f}, sarimax: {mpe(y_true, y_pred): 0.2f}, knn: {mpe(y_true, knn_preds)}')
        plt.show()
        p = os.path.join(save_dir, f'{detid}.png')
        plt.savefig(p)


if __name__ == "__main__":
    plot_sarimax_lstm()
    from statsmodels.tsa.seasonal import seasonal_decompose

    pred_size = 100
    n = 10

    dfs = utd.get_city_dfs('paris', True, False, False)
    df_traffic = dfs.traffic_df
    detids = df_traffic.detid.unique().tolist()

    for detid in tqdm(np.random.choice(detids, replace=False, size=n), total=n):
        tmp_df = df_traffic.loc[df_traffic.detid == detid]
        flow = tmp_df.flow
        train_flow = flow.iloc[:-pred_size]
        test_flow = flow.iloc[-pred_size:]

        sarima = SARIMAX(train_flow, order=(2, 0, 2), seasonal_order=(1, 1, 1, 24))
        fitted_model = sarima.fit()

        y_pred = fitted_model.forecast(pred_size)
        y_true = test_flow.values
        score = mpe(y_true, y_pred)


    # plot_sarimax_lstm()
    # for x in l:
    #     print(np.mean(x))


    # pred_size = 100
    # n = 5
    #
    # dfs = utd.get_city_dfs('paris', True, False, False)
    # df_traffic = dfs.traffic_df
    # detids = df_traffic.detid.unique().tolist()
    # # df_traffic = df_traffic.loc[df_traffic.detid == 1949][['day', 'interval', 'flow']]
    # # flow = df_traffic.flow
    # # flow_diff = flow.diff()
    # for detid in tqdm(np.random.choice(detids, replace=False, size=n), total=n):
    #     tmp_df = df_traffic.loc[df_traffic.detid == detid]
    #     flow = tmp_df.flow
    #     # flow_diff = flow.diff()
    #     flow_diff = flow
    #     train_flow = flow_diff.iloc[:-pred_size]
    #     test_flow = flow_diff.iloc[-pred_size:]
    #
    #     arma = partial_sarimax((1, 0, 1), (1, 1, 1, 24))
    #     arma = arma(train_flow)
    #     fitted_model = arma.fit()
    #     print(fitted_model.summary())
    #
    #     # y_pred = fitted_model.forecast(pred_size)
    #     # y_pred_diffs = fitted_model.forecast(pred_size)
    #     # y_pred = [flow.iloc[len(train_flow) - 1]]
    #     # for diff in y_pred_diffs:
    #     #     y_pred.append(y_pred[-1] + diff)
    #     # y_pred = np.array(y_pred[1:])
    #
    #     x = flow.iloc[-2 * pred_size: -pred_size].values
    #     y_true = flow.iloc[-pred_size:].values
    #
    #     f, ax = plt.subplots(figsize=(10, 5))
    #     ax.plot(np.arange(pred_size), x, 'k')
    #     ax.plot(np.arange(pred_size, 2*pred_size, 1), y_true, 'k--')
    #     ax.plot(np.arange(pred_size, 2*pred_size, 1), y_pred, 'b--')
    #     # ax.plot(np.arange(pred_size, 2*pred_size, 1), lstm_preds, 'g--')
    #     plt.title(f'AR: {mpe(y_true, y_pred): 0.2f}')
    #     plt.show()


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


