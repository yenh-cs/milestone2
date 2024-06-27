import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from Scripts.model import LSTM
from Scripts.constants import utd
from Scripts.auto_regression import partial_sarimax
from statsmodels.tsa.statespace.sarimax import SARIMAX

def ar_predict(fitted_model, pred_size, last_x):
    y_pred_diffs = fitted_model.forecast(pred_size)
    y_pred = [last_x]
    for diff in y_pred_diffs:
        y_pred.append(y_pred[-1] + diff)
    return np.array(y_pred)[1:]

def lstm_predict(model, X):
    X = torch.tensor(X[None, :], dtype=torch.float32)
    y_pred = model(X).detach().cpu().numpy().squeeze()
    return y_pred

model_weights_p = "./../Models/NoShuffle/1719332148.pt"
model_state_d = torch.load(model_weights_p, map_location=torch.device('cpu'))

lstm = LSTM(50, 100, num_layers=3)
lstm.load_state_dict(model_state_d)

pred_size = 100
n = 5

dfs = utd.get_city_dfs('paris', True, False, False)
df_traffic = dfs.traffic_df
detids = df_traffic.detid.unique().tolist()

for detid in tqdm(np.random.choice(detids, replace=False, size=n), total=n):
    tmp_df = df_traffic.loc[df_traffic.detid == detid]
    tmp_df.day = pd.to_datetime(tmp_df.day) + pd.to_timedelta(tmp_df.interval, unit='s')
    flow = tmp_df.flow

    train_flow = flow.iloc[:-pred_size]
    test_flow = flow.iloc[-pred_size:]

    flow_diff = flow.diff().dropna()
    train_flow_diff = flow_diff[:-pred_size]
    test_flow_diff = flow_diff[-pred_size:]

    sarimax = SARIMAX(train_flow_diff, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    fitted_model = sarimax.fit()

    # ar_preds = ar_predict(fitted_model, pred_size, train_flow.iloc[-1])
    y_pred_diffs = fitted_model.forecast(pred_size)
    ar_pred = [train_flow.iloc[-1]]
    for diff in y_pred_diffs:
        ar_pred.append(ar_pred[-1] + diff)

    ar_pred = np.array(ar_pred)
    lstm_preds = lstm_predict(lstm, train_flow.values)

    f, ax = plt.subplots(figsize=(10, 5))

    x = train_flow.iloc[-pred_size:]
    plt.plot(x, 'k')
    plt.plot(test_flow, 'k--')
    plt.plot(ar_pred, 'g--')
    plt.plot(lstm_preds, 'b--')
    plt.title(f'{detid}')
    plt.show()