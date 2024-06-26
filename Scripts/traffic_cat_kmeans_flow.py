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

    return city_data, cluster_centers, flow_order

def cluster_visualization(city_data, cluster_centers, flow_order):
    plt.figure(figsize=(6,6))
    colors = {'High': 'red', 'Mid': 'yellow', 'Low': 'green'}
    scatter = plt.scatter(city_data['flow'], city_data['occ'], c=city_data['category'].map(colors))

    plt.xlabel('Average Flow')
    plt.ylabel('Average Occupancy')
    plt.title('City Traffic Categorization')

    # for idx, row in city_data[city_data['category'].isin(['High', 'Mid'])].iterrows():
    #     plt.annotate(row['city'], (row['flow'], row['occ']), xytext=(5, 5),
    #                  textcoords='offset points', fontsize=8)

    # Annotate only the top 3 cities in the 'High' category
    top_high_cities = city_data[city_data['category'] == 'High'].nlargest(3, 'flow')
    for idx, row in top_high_cities.iterrows():
        plt.annotate(row['city'], (row['flow'], row['occ']), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    # Annotate all cities in the 'Mid' category
    mid_cities = city_data[city_data['category'] == 'Mid']
    for idx, row in mid_cities.iterrows():
        plt.annotate(row['city'], (row['flow'], row['occ']), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    # Create custom legend
    high_patch = mpatches.Patch(color='red', label='High')
    mid_patch = mpatches.Patch(color='yellow', label='Mid')
    low_patch = mpatches.Patch(color='green', label='Low')
    plt.legend(handles=[high_patch, mid_patch, low_patch], title="Traffic Level")

    plt.show()

    print("\nCluster Statistics:")
    for category in ['Low', 'Mid', 'High']:
        cluster_data = city_data[city_data['category'] == category]
        num_cities = len(cluster_data)
        mean_flow = cluster_data['flow'].mean()
        mean_occ = cluster_data['occ'].mean()

        print(f"\n{category} Traffic Cluster:")
        print(f"Number of cities: {num_cities}")
        print(f"Mean Flow: {mean_flow:.2f}")
        print(f"Mean Occupancy: {mean_occ:.2f}")

    for category in ['Low', 'Mid', 'High']:
        print(f"\n{category} Traffic Cities:")
        print(', '.join(city_data[city_data['category'] == category]['city'].tolist()))

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # detectors_path = './Data/detectors.csv'
    # links_path = './Data/links.csv'
    traffic_p = r'../Data/traffic.csv'
    df_traffic = pd.read_csv(traffic_p, low_memory=False)

    # Data preparation
    city_data, features = data_prep(df_traffic)

    # K-means Clustering to define low, med, high flow data
    city_data, cluster_centers, flow_order = k_means(city_data, features)

    cluster_visualization(city_data, cluster_centers, flow_order)