def data_prep(df):

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    features = ['flow', 'occ']

    # Group by city and calculate mean of features
    city_data = df.groupby('city')[features].mean().reset_index()

    return city_data, features

def dbscan_clus(city_data, features):
    # Normalize the features
    scaler = StandardScaler()
    city_data_scaled = scaler.fit_transform(city_data[features])

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    city_data['dbscan_cluster'] = dbscan.fit_predict(city_data_scaled)

    return city_data

def dbscan_cluster_visualization(city_data):
    # Create a scatter plot of the clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(city_data['flow'], city_data['speed'],
                          c=city_data['hierarchical_cluster'], cmap='viridis')
    plt.xlabel('Average Flow')
    plt.ylabel('Average Speed')
    plt.title('City Traffic Categorization (Hierarchical Clustering)')
    plt.colorbar(scatter)

    # Add annotations for cluster centers
    for cluster in range(1, 4):  # assuming 3 clusters
        cluster_points = city_data[city_data['hierarchical_cluster'] == cluster]
        center_x = cluster_points['flow'].mean()
        center_y = cluster_points['speed'].mean()
        plt.annotate(f'Cluster {cluster}', (center_x, center_y),
                     xytext=(5, 5), textcoords='offset points', fontweight='bold')

    plt.show()

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt

    # detectors_path = './Data/detectors.csv'
    # links_path = './Data/links.csv'
    traffic_p = r'../Data/traffic.csv'
    df_traffic = pd.read_csv(traffic_p, low_memory=False)

    # Data preparation
    city_data, features = data_prep(df_traffic)

    # dbscan Clustering to define low, med, high flow data
    city_data = dbscan_clus(city_data, features)

    dbscan_cluster_visualization(city_data)
