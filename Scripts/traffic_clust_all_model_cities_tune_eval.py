# function to prepare data for all the models
def data_prep(df, features):

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Downcast numeric columns to save memory
    df['flow'] = pd.to_numeric(df['flow'], downcast='float')
    df['occ'] = pd.to_numeric(df['occ'], downcast='float')

    # Group by city and calculate mean of features
    city_data = df.groupby('city')[features].mean().reset_index()
    scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(df[features])
    scaled_features = scaler.fit_transform(city_data[features])
    city_labels = city_data['city'].values

    return scaled_features, city_data, city_labels

def plot_silhouette_scores(scores, title):
    plt.figure()
    plt.plot(scores[:, 0], scores[:, 1], 'rx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(title)
    plt.show()

def plot_silhouette_distortion_scores(silhouette_scores, distortion_scores, title):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot silhouette scores with a red line
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette Score', color='red')
    ax1.plot(silhouette_scores[:, 0], silhouette_scores[:, 1], 'rx-', label='Silhouette Score')
    ax1.tick_params(axis='y', labelcolor='red')

    # Create a second y-axis for distortion scores
    ax2 = ax1.twinx()
    ax2.set_ylabel('Distortion Score', color='green')
    ax2.plot(distortion_scores[:, 0], distortion_scores[:, 1], 'gx-', label='Distortion Score')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add title and legends
    fig.suptitle(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

# Function to evaluate clustering results
def evaluate_clustering(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    return silhouette_avg, db_score

# Function for KMeans clustering with hyperparameter tuning
def tune_kmeans(X):
    silhouette_avg_scores = []
    distortion_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        distortion = kmeans.inertia_
        silhouette_avg_scores.append((k, silhouette_avg))
        distortion_scores.append((k, distortion))
        print("tuning complete for k=", k)
    silhouette_avg_scores = np.array(silhouette_avg_scores)
    distortion_scores = np.array(distortion_scores)
    best_k = silhouette_avg_scores[np.argmax(silhouette_avg_scores[:, 1])][0]
    return best_k, silhouette_avg_scores, distortion_scores

# Function to perform GMM clustering with hyperparameter tuning
def tune_gmm(X):
    silhouette_avg_scores = []
    for k in range(2, 10):
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_avg_scores.append((k, silhouette_avg))
    silhouette_avg_scores = np.array(silhouette_avg_scores)
    best_k = silhouette_avg_scores[np.argmax(silhouette_avg_scores[:, 1])][0]
    return best_k, silhouette_avg_scores

# Function to perform DBSCAN clustering with hyperparameter tuning
def tune_dbscan(X):
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    optimal_eps = distances[int(len(distances) * 0.95)]

    silhouette_avg_scores = []
    for eps in np.linspace(optimal_eps / 2, optimal_eps * 2, 10):
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)
        if len(set(labels)) > 1:  # Valid clusters should be more than 1
            silhouette_avg = silhouette_score(X, labels)
            silhouette_avg_scores.append((eps, silhouette_avg))
    silhouette_avg_scores = np.array(silhouette_avg_scores)
    best_eps = silhouette_avg_scores[np.argmax(silhouette_avg_scores[:, 1])][0]
    return best_eps, silhouette_avg_scores

# Function to plot clustering results
def plot_clusters(df, labels, title):
    plt.figure(figsize=(6,6))
    colors = {'High': 'red', 'Mid': 'yellow', 'Low': 'green'}

    unique_labels = np.unique(labels)
    categories = ['Low', 'Mid', 'High']

    for k in unique_labels:
        class_member_mask = (labels == k)
        col = colors[categories[k]]
        plt.plot(city_data[class_member_mask]['flow'], city_data[class_member_mask]['occ'], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=10)

    # Annotate only the top 3 cities in the 'High' category
    top_high_cities = city_data[city_data['category'] == 'High'].nlargest(3, 'flow')
    for idx, row in top_high_cities.iterrows():
        plt.annotate(row['city'], (row['flow'], row['occ']), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    # Annotate only the top 3 cities in the 'Mid' category
    top_mid_cities = city_data[city_data['category'] == 'Mid'].nlargest(3, 'flow')
    for idx, row in top_mid_cities.iterrows():
        plt.annotate(row['city'], (row['flow'], row['occ']), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    # Create custom legend
    high_patch = mpatches.Patch(color='red', label='High')
    mid_patch = mpatches.Patch(color='yellow', label='Mid')
    low_patch = mpatches.Patch(color='green', label='Low')
    plt.legend(handles=[high_patch, mid_patch, low_patch], title="Traffic Level", loc='upper right')

    plt.xlabel('Flow')
    plt.ylabel('Occupancy')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    import sklearn
    sklearn.set_config(working_memory=2048)
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.neighbors import NearestNeighbors

    traffic_p = r'../Data/traffic.csv'
    needed_columns = ['flow', 'occ', 'city']
    df_traffic = pd.read_csv(traffic_p, low_memory=False)[needed_columns]

    features = ['flow', 'occ']

    # Data preparation
    X, city_data, city_labels = data_prep(df_traffic, features)

    # KMeans Clustering
    best_k, kmeans_silhouette_scores, kmeans_distortion_scores = tune_kmeans(X)
    kmeans = KMeans(n_clusters=int(best_k), random_state=42).fit(X)
    kmeans_labels = kmeans.labels_
    plot_silhouette_distortion_scores(kmeans_silhouette_scores, kmeans_distortion_scores, "KMeans Silhouette vs. Distortion Scores")
    city_data['category'] = ['Low' if lbl == 0 else 'Mid' if lbl == 1 else 'High' for lbl in kmeans_labels]
    plot_clusters(X, kmeans_labels, "KMeans Clustering Results")

    # GMM Clustering
    best_k, gmm_silhouette_scores = tune_gmm(X)
    gmm = GaussianMixture(n_components=int(best_k), random_state=42).fit(X)
    gmm_labels = gmm.predict(X)
    plot_silhouette_scores(gmm_silhouette_scores, "GMM Silhouette Scores")
    city_data['category'] = ['Low' if lbl == 0 else 'Mid' if lbl == 1 else 'High' for lbl in gmm_labels]
    plot_clusters(city_data, gmm_labels, "GMM Clustering Results")

    # DBSCAN Clustering
    best_eps, dbscan_silhouette_scores = tune_dbscan(X)
    dbscan = DBSCAN(eps=best_eps, min_samples=5).fit(X)
    dbscan_labels = dbscan.labels_
    plot_silhouette_scores(dbscan_silhouette_scores, "DBSCAN Silhouette Scores")
    city_data['category'] = ['Low' if lbl == 0 else 'Mid' if lbl == 1 else 'High' for lbl in dbscan_labels]
    plot_clusters(city_data, dbscan_labels, "DBSCAN Clustering Results")

    # Performance Evaluation
    kmeans_silhouette, kmeans_db = evaluate_clustering(X, kmeans_labels)
    gmm_silhouette, gmm_db = evaluate_clustering(X, gmm_labels)
    dbscan_silhouette, dbscan_db = evaluate_clustering(X, dbscan_labels)
    # # spectral_silhouette, spectral_db = evaluate_clustering(X, spectral_labels)

    # Print evaluation results
    print(f"KMeans - Silhouette: {kmeans_silhouette}, Davies-Bouldin: {kmeans_db}")
    print(f"GMM - Silhouette: {gmm_silhouette}, Davies-Bouldin: {gmm_db}")
    print(f"DBSCAN - Silhouette: {dbscan_silhouette}, Davies-Bouldin: {dbscan_db}")
    # print(f"Spectral Clustering - Silhouette: {spectral_silhouette}, Davies-Bouldin: {spectral_db}")

    # Visualize and compare all clustering methods
    methods = ['KMeans', 'GMM', 'DBSCAN']
    silhouette_scores = [kmeans_silhouette, gmm_silhouette, dbscan_silhouette]
    db_scores = [kmeans_db, gmm_db, dbscan_db]

    # Convert lists to numpy arrays for easier plotting
    methods = np.array(methods)
    silhouette_scores = np.array(silhouette_scores)
    db_scores = np.array(db_scores)

    # Create a line chart
    plt.figure(figsize=(6, 6))
    plt.plot(methods, silhouette_scores, 'g-', marker='o', label='Silhouette Score')
    plt.plot(methods, db_scores, 'r-', marker='o', label='Davies-Bouldin Score')
    plt.xlabel('Clustering Method')
    plt.ylabel('Score')
    plt.title('Comparison of Clustering Method Scores')
    plt.legend()
    plt.show()