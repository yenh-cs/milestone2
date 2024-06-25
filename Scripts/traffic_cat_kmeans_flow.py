def data_prep(df):

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    features = ['flow', 'occ']

    # Group by city and calculate mean of features
    city_data = df.groupby('city')[features].mean().reset_index()

    return city_data, features

def k_means(city_data, features):
    # Normalize the features
    scaler = StandardScaler()
    city_data_scaled = scaler.fit_transform(city_data[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    city_data['cluster'] = kmeans.fit_predict(city_data_scaled)

    # Map clusters to categories
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    flow_order = cluster_centers[:, 0].argsort()
    categories = {flow_order[0]: 'Low', flow_order[1]: 'Mid', flow_order[2]: 'High'}
    city_data['category'] = city_data['cluster'].map(categories)

    return city_data, cluster_centers, flow_order, categories

def cluster_visualization(city_data, cluster_centers, flow_order, categories):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(city_data['flow'], city_data['occ'], c=city_data['cluster'], cmap='viridis')
    plt.xlabel('Average Flow')
    plt.ylabel('Average Occupancy')
    plt.title('City Traffic Categorization')
    plt.colorbar(scatter)
    for i, category in enumerate(['Low', 'Mid', 'High']):
        center = cluster_centers[flow_order[i]]
        plt.annotate(category, xy=(center[0], center[2]), xytext=(5, 5),
                     textcoords='offset points', fontweight='bold')
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # detectors_path = './Data/detectors.csv'
    # links_path = './Data/links.csv'
    traffic_p = r'../Data/traffic.csv'
    df_traffic = pd.read_csv(traffic_p, low_memory=False)

    # Data preparation
    city_data, features = data_prep(df_traffic)

    # K-means Clustering to define low, med, high flow data
    city_data, cluster_centers, flow_order, categories = k_means(city_data, features)

    cluster_visualization(city_data, cluster_centers, flow_order, categories)