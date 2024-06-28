def data_prep(df):

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # selecting key features that may account for better traffic management
    features = ['flow', 'occ', 'interval','speed']

    # # Group by city and calculate mean of features
    # city_data = df.groupby('city')[features].mean().reset_index()

    return df, features

def rfc_predict(df_traffic, features):

    # define X
    X = df_traffic[features]

    # define target Variable: using binary classification based on flow for now
    df_traffic['target'] = np.where(df_traffic['flow'] > df_traffic['flow'].median(), 1, 0)
    y = df_traffic['target']

    # Splitting data between training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardizing Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Feature Importance
    feature_importance = model.feature_importances_
    feature_names = features

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Predicting Traffic Management Efficiency')
    plt.show()

    # Print Feature Importance
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")

    return

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    import matplotlib.pyplot as plt

    # loading the data
    traffic_p = r'../Data/traffic.csv'
    df_traffic = pd.read_csv(traffic_p, low_memory=False)

    # preparing data
    df_traffic, features = data_prep(df_traffic)

    # using random forest classifiers to identify which features are important
    # for better traffic management
    rfc_predict(df_traffic, features)
