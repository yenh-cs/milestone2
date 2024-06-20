"""
Estimate an AR-X model using Conditional Maximum Likelihood (OLS).
"""
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

from Scripts.constants import utd
from Scripts.datasets import UTDCityDataset

dfs = utd.get_city_dfs('paris', True, False, False)
df_traffic = dfs.traffic_df

data = df_traffic.flow.diff().dropna()

xs = sm.tsa.stattools.pacf(data)
xs = [(i, np.abs(x)) for i, x in enumerate(xs) if i != 0]
xs = sorted(xs, key=lambda x: x[1], reverse=True)
lags = [x[0] for x in xs[:5]]

dset = UTDCityDataset('paris', 100, 50)
for i in np.random.randint(0, len(dset), 10):
    x, y = dset[i]
    model = AutoReg(x, lags)
    fitted_model = model.fit()
    y_preds = fitted_model.predict(len(x), len(x) + 50 - 1)
    plt.plot(np.arange(len(x)), x, "k")
    plt.plot(np.arange(len(x), len(x) + 50), y, "b--")
    plt.plot(np.arange(len(x), len(x) + 50), y_preds, "r--")
    plt.show()

