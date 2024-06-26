def data_prep(df):

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    features = ['flow', 'occ']

    # Group by city and calculate mean of features
    city_data = df.groupby('city')[features].mean().reset_index()

    return city_data, features

def spect_clus(city_data, features):
    # Normalize the features
    scaler = StandardScaler()
    city_data_scaled = scaler.fit_transform(city_data[features])

    num_clusters = 3
    # Apply spectral clustering
    spectral = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=42)
    city_data['spectral_cluster'] = spectral.fit_predict(city_data_scaled)

    return city_data, num_clusters

def spect_cluster_visualization(city_data, num_clusters):
    # Define colors for clusters
    colors = ['green', 'yellow', 'red']

    # Sort clusters by average flow to assign colors
    cluster_avg_flow = city_data.groupby('spectral_cluster')['flow'].mean().sort_values()
    color_map = dict(zip(cluster_avg_flow.index, colors))

    # Create a 2D scatter plot
    plt.figure(figsize=(6, 6))

    for cluster in range(num_clusters):
        cluster_data = city_data[city_data['spectral_cluster'] == cluster]
        plt.scatter(cluster_data['flow'], cluster_data['occ'],
                    c=color_map[cluster], label=f'Cluster {cluster}', s=50)  # Increased point size

    plt.xlabel('Average Flow')
    plt.ylabel('Average Occupancy')
    plt.title('City Traffic Categorization (Spectral Clustering)')
    plt.legend(['Low Traffic', 'Medium Traffic', 'High Traffic'])

    # Label city names only for medium and high traffic cities
    for i, row in city_data.iterrows():
        if row['spectral_cluster'] in [1, 2]:  # Medium and High traffic clusters
            plt.annotate(row['city'], (row['flow'], row['occ']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8,
                         va='bottom', ha='left')

    plt.tight_layout()
    plt.show()

    # Print cluster statistics
    print(city_data.groupby('spectral_cluster')[features].mean())

    # Print the number of cities in each cluster
    print(city_data['spectral_cluster'].value_counts())

    # Print city names in each cluster with the cluster name
    for cluster in range(num_clusters):
        print(f"\nCluster {cluster} ({'Low' if cluster == 0 else 'Medium' if cluster == 1 else 'High'} Traffic):")
        print(city_data[city_data['spectral_cluster'] == cluster]['city'].tolist())

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import SpectralClustering
    import matplotlib.pyplot as plt

    # detectors_path = './Data/detectors.csv'
    # links_path = './Data/links.csv'
    traffic_p = r'../Data/traffic.csv'
    df_traffic = pd.read_csv(traffic_p, low_memory=False)

    # Data preparation
    city_data, features = data_prep(df_traffic)

    # dbscan Clustering to define low, med, high flow data
    city_data, num_clusters = spect_clus(city_data, features)

    spect_cluster_visualization(city_data, num_clusters)
