""" Random Plots made for paper """
import os
import pickle

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Scripts.model import LSTM
from Scripts.constants import utd
from Scripts.eval_supervised import lstm_predict, knn_predict


def save_vis(f, p=None):
    """ saves vis or shows """
    fig, ax = f()
    plt.show()
    if p is not None:
        fig.tight_layout()
        fig.savefig(p)


def acf():
    """ autocorrelation plot """
    dfs = utd.get_city_dfs('paris', True, False, False)
    df_traffic = dfs.traffic_df
    df_traffic = df_traffic.loc[df_traffic.detid == 1949][['day', 'interval', 'flow']]

    flow = df_traffic.flow.iloc[:500]
    lags = [*np.arange(25)]
    fig = plot_acf(flow, lags=lags)
    return fig, None


def pacf():
    """ partial autocorrelation plot """
    dfs = utd.get_city_dfs('paris', True, False, False)
    df_traffic = dfs.traffic_df
    df_traffic = df_traffic.loc[df_traffic.detid == 1949][['day', 'interval', 'flow']]

    flow = df_traffic.flow.iloc[:500]
    lags = [*np.arange(25)]
    fig = plot_pacf(flow, lags=lags)
    return fig, None


def seasonal_decomp():
    """ seasonal decomposition plot """
    dfs = utd.get_city_dfs('paris', True, False, False)
    df_traffic = dfs.traffic_df
    df_traffic = df_traffic.loc[df_traffic.detid == 1949][['day', 'interval', 'flow']]

    flow = df_traffic.flow.iloc[:500]
    decomp = seasonal_decompose(flow, model='additive', period=24)
    f = decomp.plot()
    return f, None


def flow_hrs(city='paris'):
    """ Flow over time for city """
    dfs = utd.get_city_dfs(city, True, False, False)
    df_traffic = dfs.traffic_df
    df_traffic = df_traffic.loc[df_traffic.detid == 1949][['day', 'interval', 'flow']]

    flow = df_traffic.flow.iloc[:73]
    flow = flow.diff(24)

    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(0, 73), flow.values)
    xticks = np.arange(0, 80, 8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    # ax.plot(flow.values)
    ax.set_xlabel('Hours')
    ax.set_ylabel('Traffic Flow')
    plt.title('Paris Traffic Flow Seasonally Adjusted')
    plt.savefig('paris_traffic_flow.png')

    return f, ax


def lstm_loss():
    """ LSTM loss """
    df = pd.read_csv("./../Data/lstm_train.csv")
    plt.plot(np.arange(1, 31), df.train_loss, 'g')
    plt.plot(np.arange(1, 31), df.val_loss, 'r')

    green_patch = mpatches.Patch(color='green', label='Train')
    red_patch = mpatches.Patch(color='red', label="Validation")
    plt.legend(handles=[green_patch, red_patch])
    plt.title("LSTM Loss Curves")


def results_to_stats(save_p: str):
    """ model metrics to summary stats """
    p = r'./../Data/toronto_results.csv'
    df = pd.read_csv(p)
    metrics = ['sarima_mpe', 'sarima_rmse', 'sarima_mae', 'knn_mpe', 'knn_rmse', 'knn_mae', 'lstm_mpe', 'lstm_rmse',
               'lstm_mae']

    results = {}

    for metric in metrics:
        results[metric] = {
            'mean': df[metric].mean(),
            'std': df[metric].std(),
            'median': df[metric].median()
        }

    data = {
        ('MPE', 'Mean'): [results['sarima_mpe']['mean'], results['knn_mpe']['mean'], results['lstm_mpe']['mean']],
        ('MPE', 'Median'): [results['sarima_mpe']['median'], results['knn_mpe']['median'],
                            results['lstm_mpe']['median']],
        ('MPE', 'Std'): [results['sarima_mpe']['std'], results['knn_mpe']['std'], results['lstm_mpe']['std']],

        ('MAE', 'Mean'): [results['sarima_mae']['mean'], results['knn_mae']['mean'], results['lstm_mae']['mean']],
        ('MAE', 'Median'): [results['sarima_mae']['median'], results['knn_mae']['median'],
                            results['lstm_mae']['median']],
        ('MAE', 'Std'): [results['sarima_mae']['std'], results['knn_mae']['std'], results['lstm_mae']['std']],

        ('RMSE', 'Mean'): [results['sarima_rmse']['mean'], results['knn_rmse']['mean'], results['lstm_rmse']['mean']],
        ('RMSE', 'Median'): [results['sarima_rmse']['median'], results['knn_rmse']['median'],
                             results['lstm_rmse']['median']],
        ('RMSE', 'Std'): [results['sarima_rmse']['std'], results['knn_rmse']['std'], results['lstm_rmse']['std']],

    }

    index = ['SARIMA', 'KNN', 'LSTM']
    df = pd.DataFrame(data, index=index)
    df.to_csv(save_p, index_label='method')


def mpe_kdes():
    """ Mean Percentage Error KDEs for each model """
    p = r'./../Data/toronto_results.csv'
    df = pd.read_csv(p)
    df = df[[x for x in df.columns if "mpe" in x]]

    df = df.dropna()
    df = np.clip(df, -1, 1)

    df = df.rename(columns={k: k.replace("_mpe", "") for k in df.columns})
    val_col = "Mean Percentage Error"
    df = df.melt(var_name='method', value_name=val_col)
    df.method = df.method.replace({'sarima': "SARIMA", "knn": "KNN", "lstm": "LSTM"})
    palette = {'SARIMA': 'red', 'KNN': 'gold', 'LSTM': 'green'}

    f, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(df, x=val_col, hue='method', ax=ax, palette=palette)
    f.tight_layout()
    return f, ax


def grid_search_heat_map():
    """ Grid Search Heat Map """
    p = r"./../Data/knn_grid_search.csv"

    df = pd.read_csv(p)
    df = df[['param_n_neighbors', 'param_p', 'param_weights', 'mean_test_score', 'std_test_score']]

    pivot_df = df.pivot_table(index=['param_n_neighbors', 'param_p'], columns='param_weights',
                                      values='mean_test_score')

    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap="Spectral", fmt=".3f", ax=ax)
    plt.title('Grid Search Mean Test Scores')
    plt.xlabel('Weights')
    plt.ylabel('N Neighbors-p')
    return f, ax


def plot_sarimax_knn_lstm(save_dir: str, city="paris", pred_size=100, n=10):
    """
    Plots sarimax, lnn, and lstm series prediction
    Args:
        save_dir: where to save images
        city: utd city
        pred_size: number of prediction
        n: number of detid to predict

    Returns:
        None, saves images and shows image
    """
    model_weights_p = "./../Models/NoShuffle/1719332148.pt"
    model_state_d = torch.load(model_weights_p, map_location=torch.device('cpu'))
    lstm = LSTM(50, pred_size, num_layers=3)
    lstm.load_state_dict(model_state_d)

    knn_p = "./../Models/knn.pkl"
    with open(knn_p, 'rb') as f:
        knn = pickle.load(f)

    dfs = utd.get_city_dfs(city, True, False, False)
    df_traffic = dfs.traffic_df
    detids = df_traffic.detid.unique().tolist()

    for detid in tqdm(np.random.choice(detids, replace=False, size=n), total=n):
        tmp_df = df_traffic.loc[df_traffic.detid == detid]
        flow = tmp_df.flow
        train_flow = flow.iloc[:-pred_size]
        test_flow = flow.iloc[-pred_size:]

        sarima = SARIMAX(train_flow, order=(2, 0, 2))
        fitted_model = sarima.fit()

        y_pred = fitted_model.forecast(pred_size)
        lstm_preds = lstm_predict(lstm, flow.iloc[:-pred_size].values)
        knn_preds = knn_predict(knn, flow.iloc[:-pred_size].values)

        x = flow.iloc[-2 * pred_size: -pred_size].values
        y_true = test_flow.values

        f, axes = plt.subplots(nrows=3, sharex="all", sharey='all', figsize=(10, 10))
        for ax in axes:
            ax.plot(np.arange(pred_size), x, 'k')
            ax.plot(np.arange(pred_size, 2*pred_size, 1), y_true, 'k--')
        axes[0].plot(np.arange(pred_size, 2*pred_size, 1), y_pred, 'r--')
        axes[1].plot(np.arange(pred_size, 2*pred_size, 1), knn_preds, color='gold', linestyle='--')
        axes[2].plot(np.arange(pred_size, 2*pred_size, 1), lstm_preds, 'g--')

        black_patch = mpatches.Patch(color='black', label="Truth")
        red_patch = mpatches.Patch(color='red', label="SARIMA")
        yellow_patch = mpatches.Patch(color='gold', label='KNN')
        green_patch = mpatches.Patch(color='green', label="LSTM")

        axes[0].legend(handles=[black_patch, red_patch])
        axes[1].legend(handles=[black_patch, yellow_patch])
        axes[2].legend(handles=[black_patch, green_patch])
        p = os.path.join(save_dir, f'{detid}.png')
        f.tight_layout()
        f.savefig(p)
        plt.show()
