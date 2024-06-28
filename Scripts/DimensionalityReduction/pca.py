import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns


def apply_pca(file_path, n_components):
    traffic_p = file_path + '/traffic.csv'
    detectors_p = file_path + '/detector.csv'

    # Select features for PCA
    df_traffic = pd.read_csv(traffic_p).drop(columns=['speed', 'error']).dropna()
    df_detector = pd.read_csv(detectors_p).dropna()[['detid', 'length', 'pos']]

    df_traffic.occ = df_traffic.occ.replace(np.inf, 1E3)
    df = df_traffic.merge(df_detector, on='detid', how='outer').dropna()
    x = df[['flow', 'occ', 'length', 'pos']]
    y = df.city

    # Normalize the features
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)

    # Apply PCA
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x_normalized)


    return pca, x_pca


def biplot_pca(pca, x_pca, n_components):
    """
        Adapted from https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
        Args:
            score:
            coeff:
            labels:

        Returns:

    """
    score = x_pca
    coeff = np.transpose(pca.components_)
    labels =  ['flow', 'occ', 'length', 'pos']

    if n_components == 2:
        xs = score[:, 0]
        ys = score[:, 1]
        n = coeff.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        sns.scatterplot(x=xs * scalex, y=ys * scaley, hue=ys)
        for i in range(n):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
            if labels is None:
                plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center',
                         va='center')
            else:
                plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel("PC{}".format(1), fontweight='bold')
        plt.ylabel("PC{}".format(2), fontweight='bold')
        plt.grid()
        plt.legend().set_visible(False)
        plt.title(f'PCA Biplot of Traffic Data (n_components={n_components})', fontweight='bold')
        plt.show()
    else:
        print(f"Cannot create biplot with n_components={n_components}")


if __name__ == "__main__":
    # detectors_path = './Data/detectors.csv'
    # links_path = './Data/links.csv'
    # traffic_path = './Data/traffic.csv'

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Traverse up to the root directory (milestone2 in this case)
    root_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    utd_path = os.path.join(root_dir, "Data/UTD")

    # This code for looping over all cities but for the performance and demonstration purpose, we will run on 1 city only
    # cities = os.listdir(utd_path)
    # for city in tqdm(cities):
    #     if city[0] == ".":
    #         continue
    #
    #     #city folder path within UTD
    #     city_path = os.path.join(utd_path, city)
    #     combined_data_path = os.path.join(city_path, "combined_data_{city}.csv".format(city=city))
    #
    #     combined_df = load_combined_data_city(combined_data_path)
    #
    #     # print(len(combined_df))
    #     pca_df = pca(combined_df)
    #
    #     # combined_df = pd.concat([combined_df, pca_df], axis=1)
    #     # print("pca test 13 pass")
    #     #
    #     # # Explained variance
    #     # explained_variance = pca.explained_variance_ratio_
    #     # print("pca test 14 pass")
    #     # print("Explained variance by each principal component: ", explained_variance)
    #     #
    #     # # Visualize the PCA results
    #     # plot_pca_results(combined_df)

    city_path = os.path.join(utd_path, 'paris')
    for n_components in [2, 3, 4]:
        print(f"\nApplying PCA with n_components={n_components}")
        pca, x_pca = apply_pca(city_path, n_components)
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by each principal component for n_components={n_components}: ", explained_variance)
        # Visualize the PCA results
        biplot_pca(pca, x_pca, n_components)


