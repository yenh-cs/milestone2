import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def cluster_by_city(city_path):
    # Load datasets
    detectors_path = city_path + '/detector.csv'
    traffic_path = city_path + '/traffic.csv'

    df_detectors = pd.read_csv(detectors_path)
    df_traffic = pd.read_csv(traffic_path)

    # Preprocess traffic data
    df_traffic['day'] = pd.to_datetime(df_traffic['day'])
    df_traffic['day_of_week'] = df_traffic['day'].dt.dayofweek  # Monday=0, Sunday=6

    # Rename 'city' in df_traffic to 'citycode' to match with df_detectors
    df_traffic.rename(columns={'city': 'citycode'}, inplace=True)

    # Merge traffic and detector datasets on 'detid' and 'citycode'
    df_merged = pd.merge(df_traffic, df_detectors, on=['detid', 'citycode'], how='inner')

    # Select relevant columns for Clustering
    df_traffic_cluster = df_merged[
        ['day_of_week', 'interval', 'flow', 'occ', 'limit', 'lanes', 'long', 'lat']]

    # Fill missing values with 0
    df_traffic_cluster.fillna(0, inplace=True)

    # Convert interval from seconds to hours for better interpretability
    df_traffic_cluster['interval_hours'] = df_traffic_cluster['interval'] / 3600
    df_traffic_cluster.drop(columns=['interval'], inplace=True)

    # Extract relevant features for Clustering
    features = ['day_of_week', 'interval_hours', 'flow', 'occ', 'limit', 'lanes', 'long', 'lat']
    X = df_traffic_cluster[features]

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Apply KMeans Clustering
    n_clusters = 3  # Define the number of clusters (high, medium, low traffic)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_pca)

    # Add cluster labels to the original dataframe
    df_traffic_cluster['cluster'] = kmeans.labels_

    # Analyze the clusters
    cluster_summary = df_traffic_cluster.groupby('cluster').mean()

    return df_traffic_cluster, pca, X_pca


def plot_the_clusters(clusters, city, pca, X_pca):
    # Plot the clusters in PCA space
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters['cluster'], palette='viridis')
    plt.title(f'PCA Clusters for {city}', fontweight='bold')
    plt.xlabel('Principal Component 1', fontweight='bold')
    plt.ylabel('Principal Component 2', fontweight='bold')
    plt.legend(title='Cluster')
    plt.show()

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='interval_hours', y='flow', hue='cluster', data=clusters, palette='viridis')
    plt.title(f'Traffic Clusters for {city}', fontweight='bold')
    plt.xlabel('Time of Day (hours)', fontweight='bold')
    plt.ylabel('Traffic Flow (vehicles/hour)', fontweight='bold')
    plt.legend(title='Cluster')
    plt.show()

    # Additional analysis: distribution of clusters by day of the week
    plt.figure(figsize=(10, 6))
    sns.countplot(x='day_of_week', hue='cluster', data=clusters, palette='viridis')
    plt.title(f'Cluster Distribution by Day of the Week for {city}', fontweight='bold')
    plt.xlabel('Day of the Week', fontweight='bold')
    plt.ylabel('Number of Records', fontweight='bold')
    plt.legend(title='Cluster')
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Traverse up to the root directory (milestone2 in this case)
    root_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    utd_path = os.path.join(root_dir, "Data/UTD")

    cities = os.listdir(utd_path)

    # for city in tqdm(cities):
    #     if city[0] == ".":
    #         continue
    #
    #     #city folder path within UTD
    #     city_path = os.path.join(utd_path, city)
    #     # combined_data_path = os.path.join(city_path, "combined_data_{city}.csv".format(city=city))
    #
    #     # combined_df = load_combined_data_city(city_path)
    #     #
    #     clusters = cluster_by_city(city_path)
    #     plot_the_clusters(clusters)

    city = 'paris'
    city_path = os.path.join(utd_path, city)
    clusters, pca, X_pca = cluster_by_city(city_path)
    plot_the_clusters(clusters, city, pca, X_pca)
