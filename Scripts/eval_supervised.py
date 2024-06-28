"""
Multiprocessing Evaluation of Supervised Methods
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle
from typing import Dict, Callable, List

from Scripts.model import LSTM
import torch
from Scripts.auto_regression import partial_sarimax

from Scripts.constants import utd
from Scripts.misc import mpe, rmse, mae
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np


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


def sarima_fit_transform(part_sarimax, X, pred_size):
    fitted_model = part_sarimax(X).fit()
    return fitted_model.forecast(pred_size)


class ForecastProcessor:
    METRICS = {'mpe': mpe, "rmse": rmse, "mae": mae}
    def __init__(
            self,
            df_traffic: pd.DataFrame,
            lstm,
            knn,
            sarima
    ):
        self.df_traffic = df_traffic
        self.lstm = lstm
        self.knn = knn
        self.sarima = sarima
        self._i = 0

    def return_scores(self, y_true=None, y_pred=None) -> Dict[str, float]:
        if y_true is None or y_pred is None:
            return {k: np.nan for k in self.METRICS}
        return {k: fn(y_true, y_pred) for k, fn in self.METRICS.items()}

    def forecast(self, detid, pred_size=100):
        self._i += 1
        print(f'======= {self._i} =======')
        tmp_df = self.df_traffic.loc[self.df_traffic.detid == detid]
        flow = tmp_df.flow
        train_flow = flow.iloc[:-pred_size]
        y_true = flow.iloc[-pred_size:].values

        def try_predict(fn, args):
            try:
                preds = fn(*args)
            except:
                preds = None

            return preds

        knn_preds = try_predict(knn_predict, (self.knn, flow.iloc[:-pred_size].values))
        lstm_preds = try_predict(lstm_predict, (self.lstm, flow.iloc[:-pred_size].values))
        sarima_preds = try_predict(sarima_fit_transform, (self.sarima, train_flow, pred_size))

        out_d = {
            "sarima": self.return_scores(y_true, sarima_preds),
            "knn": self.return_scores(y_true, knn_preds),
            "lstm": self.return_scores(y_true, lstm_preds)
        }

        return out_d


def parallel_forecast(args):
    forecaster, detid, pred_size = args
    return forecaster.forecast(detid, pred_size)


def save_results(results: List[dict], save_p: str, **kwargs):
    df = pd.json_normalize(results, sep="_")
    for k, vals in kwargs.items():
        df[k] = vals

    df.to_csv(save_p, index_label='run')


def main(save_p: str, pred_size=100):
    model_weights_p = "./../Models/NoShuffle/1719332148.pt"
    model_state_d = torch.load(model_weights_p, map_location=torch.device('cpu'))
    lstm = LSTM(50, 100, num_layers=3)
    lstm.load_state_dict(model_state_d)

    knn_p = "./../Models/knn.pkl"
    with open(knn_p, 'rb') as f:
        knn = pickle.load(f)

    sarima = partial_sarimax((2, 0, 2), (1, 0, 1, 24))

    dfs = utd.get_city_dfs('toronto', True, False, False)
    df_traffic = dfs.traffic_df

    forecaster = ForecastProcessor(df_traffic, lstm, knn, sarima)

    detids = df_traffic.detid.unique().tolist()
    detids = np.random.choice(detids, 100)

    with Pool(cpu_count() // 2) as pool:
        results = pool.map(parallel_forecast, [(forecaster, detid, pred_size) for detid in detids])

    save_results(results, save_p, detid=detids)
