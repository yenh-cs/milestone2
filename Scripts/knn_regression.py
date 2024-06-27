import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from Scripts.constants import utd
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    dfs = utd.get_city_dfs("paris")
    traffic_df = dfs.traffic_df

    detids = traffic_df.detid.unique()

    xs = []
    ys = []

    pred_len = 100

    for detid in detids:
        flow = traffic_df.loc[traffic_df.detid == detid]["flow"].values


        x = flow[-200 - 2 * pred_len: -2 * pred_len]
        y = flow[-2 * pred_len: -pred_len]

        if len(x) != 200 or len(y) != pred_len:
            continue

        xs.append(x)
        ys.append(y)


    X_train = np.array(xs)
    y_train = np.array(ys)

    knn = KNeighborsRegressor()

    params = {
        'weights': ['uniform', 'distance'],
        'n_neighbors': list(range(2, 10)),
        'p': [1., 1.5, 2., 2.5]
    }

    gs_cv = GridSearchCV(
        estimator=knn,
        param_grid=params,
        n_jobs=4,
        verbose=0
    )
    gs_cv.fit(X_train, y_train)

    data = gs_cv.cv_results_
    df = pd.DataFrame(data)
    print(df.head().to_string())
    df.to_csv("./../../Models/knn_grid_search.csv", index=False)

    knn = gs_cv.best_estimator_
    print(os.path.abspath(os.path.join(__file__, '..')))
    with open("../../Models/knn.pkl", "wb") as f:
        pickle.dump(knn, f)
