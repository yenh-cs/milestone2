import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import prince
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def load_and_preprocess(file_path):
    traffic_p = os.path.join(file_path, 'traffic.csv')
    detectors_p = os.path.join(file_path, 'detector.csv')

    df_traffic = pd.read_csv(traffic_p).drop(columns=['speed', 'error']).dropna()
    df_detector = pd.read_csv(detectors_p).dropna()[['detid', 'length', 'pos']]

    df_traffic.occ = df_traffic.occ.replace(np.inf, 1E3)
    df = df_traffic.merge(df_detector, on='detid', how='outer').dropna()
    x = df[['flow', 'occ', 'length', 'pos']]
    y = df['city']

    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)

    return x_normalized, y

def apply_pca(x_normalized):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_normalized)
    explained_variance = pca.explained_variance_ratio_

    return x_pca, explained_variance

def apply_tsne(x_normalized):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    x_tsne = tsne.fit_transform(x_normalized)

    return x_tsne

def apply_famd(df):
    famd = prince.FAMD(n_components=2, random_state=42)
    famd = famd.fit(df)
    x_famd = famd.row_coordinates(df)

    explained_variance = famd.eigenvalues_summary
    return x_famd, explained_variance

def plot_results(x_reduced, y, method):
    df_reduced = pd.DataFrame(data=x_reduced, columns=['Dim1', 'Dim2'])
    df_reduced['city'] = y.values

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Dim1', y='Dim2', hue='city', data=df_reduced, palette='viridis')
    plt.title(f'{method} of Traffic Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='City')
    plt.show()

def evaluate_clustering(x_reduced):
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(x_reduced)

    silhouette_avg = silhouette_score(x_reduced, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(x_reduced, cluster_labels)

    return silhouette_avg, davies_bouldin_avg

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    utd_path = os.path.join(root_dir, "Data/UTD")

    city_path = os.path.join(utd_path, 'paris')
    x_normalized, y = load_and_preprocess(city_path)

    # Apply PCA
    x_pca, pca_explained_variance = apply_pca(x_normalized)
    print("PCA Explained Variance:", pca_explained_variance)
    # plot_results(x_pca, y, 'PCA')
    pca_silhouette, pca_davies_bouldin = evaluate_clustering(x_pca)

    # Apply t-SNE
    x_tsne = apply_tsne(x_normalized)
    # plot_results(x_tsne, y, 't-SNE')
    tsne_silhouette, tsne_davies_bouldin = evaluate_clustering(x_tsne)

    # Apply FAMD
    df = pd.concat([pd.DataFrame(x_normalized, columns=['flow', 'occ', 'length', 'pos']), y.reset_index(drop=True)], axis=1)
    x_famd, famd_explained_variance = apply_famd(df)
    print("FAMD Explained Variance:", famd_explained_variance)
    # plot_results(x_famd, y, 'FAMD')
    famd_silhouette, famd_davies_bouldin = evaluate_clustering(x_famd)

    # Print clustering evaluation metrics
    print(f"PCA Silhouette Score: {pca_silhouette}, Davies-Bouldin Score: {pca_davies_bouldin}")
    print(f"t-SNE Silhouette Score: {tsne_silhouette}, Davies-Bouldin Score: {tsne_davies_bouldin}")
    print(f"FAMD Silhouette Score: {famd_silhouette}, Davies-Bouldin Score: {famd_davies_bouldin}")
