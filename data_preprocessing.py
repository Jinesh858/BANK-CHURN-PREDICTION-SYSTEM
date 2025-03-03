import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # For saving models

def preprocess_data(file_path):
    """Loads, preprocesses, and clusters the dataset."""
    
    print("📥 Loading dataset...")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("❌ Error: File not found. Please provide a valid file path.")
        return None

    # Drop non-essential columns
    print("🗑️ Dropping unnecessary columns...")
    data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True, errors='ignore')

    # Handle missing values (fill with median)
    if data.isnull().sum().sum() > 0:
        print("⚠️ Missing values detected! Filling with median...")
        data.fillna(data.median(), inplace=True)

    # Encode categorical data
    print("🔄 Encoding categorical variables...")
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

    # Define features and target
    X = data.drop(columns=['Exited'])
    y = data['Exited']

    # Scale features
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-Means clustering
    print("🔢 Performing K-Means clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Save the updated dataframe with clusters
    output_file = 'churn_with_clusters.csv'
    print(f"💾 Saving processed data to {output_file}...")
    data.to_csv(output_file, index=False)

    # Save scaler and KMeans model
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")
    print("✅ Data preprocessing complete!")

    return X_scaled, y, scaler, data, kmeans

# Run preprocessing and save the processed data with clusters
if __name__ == "__main__":
    preprocess_data('churn.csv')
