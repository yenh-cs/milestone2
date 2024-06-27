import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess(file_path):
    traffic_p = os.path.join(file_path, 'traffic.csv')
    detectors_p = os.path.join(file_path, 'detector.csv')

    # Select features for t-SNE
    df_traffic = pd.read_csv(traffic_p).drop(columns=['speed', 'error']).dropna()
    df_detector = pd.read_csv(detectors_p).dropna()[['detid', 'length', 'pos']]

    df_traffic.occ = df_traffic.occ.replace(np.inf, 1E3)
    df = df_traffic.merge(df_detector, on='detid', how='outer').dropna()

    x = df[['flow', 'occ', 'length', 'pos']]
    y = df['city']

    # Normalize the features
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)

    return x_normalized, y


def manual_tsne_tuning(x_normalized):
    param_grid = {
        'perplexity': [5, 30, 50],
        'n_iter': [300, 500, 1000],
        'learning_rate': [10, 100, 200]
    }

    best_tsne = None
    best_score = float('inf')  # For t-SNE, we often use the Kullback-Leibler divergence as a "score"
    best_params = {}

    for perplexity in param_grid['perplexity']:
        for n_iter in param_grid['n_iter']:
            for learning_rate in param_grid['learning_rate']:
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate,
                            random_state=42)
                x_tsne = tsne.fit_transform(x_normalized)

                # Use the KL divergence as a score, lower is better
                score = tsne.kl_divergence_

                if score < best_score:
                    best_score = score
                    best_tsne = tsne
                    best_params = {
                        'perplexity': perplexity,
                        'n_iter': n_iter,
                        'learning_rate': learning_rate
                    }

    print("Best parameters found: ", best_params)
    return best_tsne, best_tsne.fit_transform(x_normalized)


def plot_tsne_results(x_tsne, y):
    tsne_df = pd.DataFrame(data=x_tsne, columns=['Dim1', 'Dim2'])
    tsne_df['city'] = y.values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Dim1', y='Dim2', hue='city', data=tsne_df, palette='viridis')
    plt.title('t-SNE of Traffic Data', fontweight='bold')
    plt.xlabel('Dimension 1', fontweight='bold')
    plt.ylabel('Dimension 2', fontweight='bold')
    plt.legend(title='City')
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    utd_path = os.path.join(root_dir, "Data/UTD")

    city_path = os.path.join(utd_path, 'paris')
    x_normalized, y = load_and_preprocess(city_path)

    print(f"x_normalized shape: {x_normalized.shape}")
    print(f"y shape: {y.shape}")

    best_tsne, x_tsne = manual_tsne_tuning(x_normalized)

    plot_tsne_results(x_tsne, y)
