import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cross_decomposition

from Scripts.constants import utd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import zscore
import seaborn as sns
from sklearn.cluster import KMeans
import sys
import prince

def famd():

    traffic_p = r'../Data/traffic_group.csv'
    detectors_p = r'../Data/detectors.csv'

    df_traffic = pd.read_csv(traffic_p).drop(columns=['speed', 'error']).dropna()
    df_detector = pd.read_csv(detectors_p)
    df_detector['city'] = df_detector.citycode
    df_detector = df_detector.drop(columns=['citycode'])

    df = df_traffic.merge(df_detector, on=['city', "detid"], how='outer')
    df = df.drop(columns=["detid", "road", "linkid"]).set_index('city')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    famd = prince.FAMD(5)

    famd.fit(df)
    plot = famd.plot(
        df,
        x_component=0,
        y_component=1,
        color_rows_by='city:N'
    )
    plot.save("../Data/chart.html")
    print(famd.eigenvalues_summary)

def pca():
    traffic_p = r'../Data/traffic_group.csv'
    detectors_p = r'../Data/detectors.csv'

    df_traffic = pd.read_csv(traffic_p).drop(columns=['speed', 'error']).dropna()
    df_detector = pd.read_csv(detectors_p).dropna()[['detid', 'length', 'pos']]

    df_traffic.occ = df_traffic.occ.replace(np.inf, 1E3)
    df = df_traffic.merge(df_detector, on='detid', how='outer').dropna()
    x = df[['flow', 'occ', 'length', 'pos']]
    y = df.city

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    pca = PCA()
    x_new = pca.fit_transform(x)
    print(pca.explained_variance_ratio_)

    def biplot(score,coeff,labels=None):
        """
        Adapted from https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
        Args:
            score:
            coeff:
            labels:

        Returns:

        """
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        sns.scatterplot(x=xs * scalex, y=ys * scaley, hue=y)
        # plt.scatter(xs * scalex,ys * scaley, c = y)
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()


    biplot(x_new, np.transpose(pca.components_), ['flow', 'occ', 'length', 'posÍ'])
    plt.legend().set_visible(False)
    plt.show()

famd()