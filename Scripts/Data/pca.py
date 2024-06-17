import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px


def load_combined_data(detectors_path, links_path, traffic_path):
    df_detec = pd.read_csv(detectors_path)
    df_link = pd.read_csv(links_path)
    df_traff = pd.read_csv(traffic_path)

    # Aggregate the flow data to get the mean flow per detector
    df_traff_agg = df_traff.groupby('detid')[['flow', 'occ', 'speed']].mean().reset_index()

    # Merge the dataframes
    df = df_traff_agg.merge(df_detec, on='detid', how='outer')
    df = df.dropna(subset=['lat', 'long', 'flow'])  # Ensure latitude, longitude, and flow are present

    # Merge with links data
    df = df.merge(df_link, on='linkid', how='left')

    return df


def preprocess_data(df):
    # Fill missing values for other features if necessary
    df = df.fillna(0)
    return df


def plot_pca_results(df):
    fig = px.scatter(df, x='PC1', y='PC2', color='city', hover_data=['detid', 'flow', 'lat', 'long'])
    fig.update_layout(title='PCA of Traffic Data Across Cities',
                      xaxis_title='Principal Component 1',
                      yaxis_title='Principal Component 2')
    fig.show()


if __name__ == "__main__":
    detectors_path = './Data/detectors.csv'
    links_path = './Data/links.csv'
    traffic_path = './Data/traffic.csv'

    combined_df = load_combined_data(detectors_path, links_path, traffic_path)
    combined_df = preprocess_data(combined_df)

    # Select features for PCA
    features = combined_df.columns.difference(['detid', 'city'])  # Exclude non-numeric and identifier columns
    x = combined_df[features].values

    # Standardize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Apply PCA
    pca = PCA(n_components=2)  # Adjust n_components as needed
    principal_components = pca.fit_transform(x_scaled)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    combined_df = pd.concat([combined_df, pca_df], axis=1)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance by each principal component: ", explained_variance)

    # Visualize the PCA results
    plot_pca_results(combined_df)
