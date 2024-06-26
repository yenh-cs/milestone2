import os
import pandas as pd
import numpy as np
import prince
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess(file_path):
    traffic_p = os.path.join(file_path, 'traffic.csv')
    detectors_p = os.path.join(file_path, 'detector.csv')

    df_traffic = pd.read_csv(traffic_p).drop(columns=['speed', 'error']).dropna()
    df_detector = pd.read_csv(detectors_p).dropna()[['detid', 'length', 'pos']]

    df_traffic.occ = df_traffic.occ.replace(np.inf, 1E3)
    df = df_traffic.merge(df_detector, on='detid', how='outer').dropna()

    return df


def apply_famd(df):
    # Identify categorical and numerical columns
    categorical_cols = ['detid', 'city']
    numerical_cols = ['flow', 'occ', 'length', 'pos']

    famd = prince.FAMD(n_components=2, random_state=42)
    famd = famd.fit(df[categorical_cols + numerical_cols])

    famd_df = famd.row_coordinates(df[categorical_cols + numerical_cols])
    famd_df.columns = ['Dim1', 'Dim2']
    famd_df['city'] = df['city'].values

    return famd, famd_df


def plot_famd_results(famd_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Dim1', y='Dim2', hue='city', data=famd_df, palette='viridis')
    plt.title('FAMD of Traffic Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='City')
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    utd_path = os.path.join(root_dir, "Data/UTD")

    city_path = os.path.join(utd_path, 'paris')
    df = load_and_preprocess(city_path)

    famd, famd_df = apply_famd(df)

    plot_famd_results(famd_df)
