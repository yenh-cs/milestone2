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

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    city_data['dbscan_cluster'] = dbscan.fit_predict(city_data_scaled)

    return city_data

def dbscan_cluster_visualization(city_data):
    # Define colors for clusters
    colors = ['green', 'yellow', 'red']

    # Sort clusters by average flow to assign colors
    cluster_avg_flow = city_data.groupby('dbscan_cluster')['flow'].mean().sort_values()
    color_map = dict(zip(cluster_avg_flow.index, colors))

    # Create a scatter plot
    plt.figure(figsize=(6, 6))

    for cluster, color in color_map.items():
        cluster_data = city_data[city_data['dbscan_cluster'] == cluster]
        plt.scatter(cluster_data['flow'], cluster_data['occ'], c=color, label=f'Cluster {cluster}')

    plt.xlabel('Average Flow')
    plt.ylabel('Average Occupancy')
    plt.title('City Traffic Categorization (DBSCAN)')
    plt.legend(['Low Traffic', 'Medium Traffic', 'High Traffic'])

    # Label city names only for medium and high traffic cities
    lowest_cluster = cluster_avg_flow.index[0]
    for i, row in city_data.iterrows():
        if row['dbscan_cluster'] != lowest_cluster:
            plt.annotate(row['city'], (row['flow'], row['occ']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Print cluster statistics
    print("Cluster Statistics:")
    print(city_data.groupby('dbscan_cluster')[features].mean())

    # Print the number of cities in each cluster
    print("\nNumber of cities in each cluster:")
    print(city_data['dbscan_cluster'].value_counts())

    # Print list of cities in each cluster
    print("\nCities in each cluster:")
    for cluster, color in zip(cluster_avg_flow.index, ['Low', 'Medium', 'High']):
        print(f"\n{color} Traffic Cities (Cluster {cluster}):")
        cities = city_data[city_data['dbscan_cluster'] == cluster]['city'].tolist()
        print(", ".join(cities))


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
