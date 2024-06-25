def data_prep(df):

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    features = ['flow', 'occ']

    # Group by city and calculate mean of features
    city_data = df.groupby('city')[features].mean().reset_index()

    return city_data, features

def hier_clus(city_data, features):
    # Normalize the features
    scaler = StandardScaler()
    city_data_scaled = scaler.fit_transform(city_data[features])

    # Perform hierarchical clustering
    linked = linkage(city_data_scaled, method='ward')

    # Create the dendrogram plot
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=city_data.index,
               distance_sort='descending',
               show_leaf_counts=True)

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

    # Cut the dendrogram into 3 clusters
    city_data['hierarchical_cluster'] = fcluster(linked, t=3, criterion='maxclust')

    return city_data

def hie_cluster_visualization(city_data, cluster_centers, flow_order, categories):
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
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.cluster.hierarchy import fcluster
    import matplotlib.pyplot as plt

    # detectors_path = './Data/detectors.csv'
    # links_path = './Data/links.csv'
    traffic_p = r'../Data/traffic.csv'
    df_traffic = pd.read_csv(traffic_p, low_memory=False)

    # Data preparation
    city_data, features = data_prep(df_traffic)

    # hierarchical Clustering to define low, med, high flow data
    city_data = hier_clus(city_data, features)

    hie_cluster_visualization(city_data)
