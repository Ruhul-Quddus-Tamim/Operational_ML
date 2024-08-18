from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(data):
    # Drop rows with any null values
    data = data.dropna()

    # Encode the target variable
    label_encoder = LabelEncoder()
    data['label_encoded'] = label_encoder.fit_transform(data['label'])
    labels = data['label_encoded']
    data = data.drop(columns=['label', 'label_encoded'])

    # Check for and remove any columns with constant values (no variance)
    data = data.loc[:, (data != data.iloc[0]).any()]

    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, labels

def apply_pca(data, variance_threshold=0.95):
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=variance_threshold)
    data_pca = pca.fit_transform(data)

    return data_pca
