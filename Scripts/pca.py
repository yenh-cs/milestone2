import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

def load_combined_data_city(combined_data_path):
    df_combined_data = pd.read_csv(combined_data_path)
    return df_combined_data

def semicolon_process_limit(limit):
    if isinstance(limit, str) and ';' in limit:
        limits = limit.split(';')
        limits = [float(l) for l in limits]
        return sum(limits) / len(limits)
    elif isinstance(limit, (int, float)):
        return limit
    else:
        return float(limit)  # Handle case where it's a single string number

def pipes_process_limit(limit):
    try:
        if isinstance(limit, str) and '|' in limit:
            limits = limit.split('|')
            limits = [float(l) for l in limits]
            return sum(limits) / len(limits)
        elif isinstance(limit, (int, float)):
            return limit
        else:
            return float(limit)  # Handle case where it's a single string number
    except (ValueError, OverflowError) as e:
        print(f"Error processing limit value {limit}: {e}")
        return np.nan

def preprocess_data(df):
    # Fill missing values for other features if necessary
    # df['day'] = pd.to_datetime(df['day'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    df['limit'] = df['limit'].apply(semicolon_process_limit)
    df['limit'] = df['limit'].apply(pipes_process_limit)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # print(df.columns)
    return df

def PCA_analysis(df):

    # print(df.dtypes)
    # # Select features for PCA: Exclude non-numeric and identifier columns
    features = ['length', 'pos', 'limit', 'lanes', 'long', 'lat', 'interval', 'flow', 'occ', 'error', 'speed', 'order', 'piece', 'group']

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    df = df[features]
    df = preprocess_data(df)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    # Perform PCA
    pca = PCA(n_components=5)  # Number of components can be adjusted
    principal_components = pca.fit_transform(scaled_features)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

    # Display the results
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    # print("\nDataFrame with Principal Components:")
    # print(pca_df)

    return pca_df

def plot_pca_results(df):
    # print("plot pca test 2 pass")
    fig = px.scatter(df, x='PC1', y='PC2')
    # print("plot pca test 3 pass")
    fig.update_layout(title='PCA of Traffic Data',
                      xaxis_title='Principal Component 1',
                      yaxis_title='Principal Component 2')
    # print("plot pca test 4 pass")
    fig.show()

if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # detectors_path = './Data/detectors.csv'
    # links_path = './Data/links.csv'
    # traffic_path = './Data/traffic.csv'

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Traverse up to the root directory (milestone2 in this case)
    root_dir = os.path.abspath(os.path.join(script_dir, '../'))
    utd_path = os.path.join(root_dir, "Data/UTD")

    cities = os.listdir(utd_path)

    for city in tqdm(cities):
        if city[0] == ".":
            continue

        print(city)

        #city folder path within UTD
        city_path = os.path.join(utd_path, city)
        combined_data_path = os.path.join(city_path, "combined_data_{city}.csv".format(city=city))

        combined_df = load_combined_data_city(combined_data_path)

        # print(len(combined_df))
        pca_df = PCA_analysis(combined_df)

        # combined_df = pd.concat([combined_df, pca_df], axis=1)
        # print("pca test 13 pass")

        # Visualize the PCA results
        plot_pca_results(pca_df)
