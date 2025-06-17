import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

nltk.download('stopwords')


class SentimentDataPreprocessor:
    def __init__(self, max_features=3000, convert_to_dense=False):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.convert_to_dense = convert_to_dense

    def clean_text(self, text):
        """Clean and preprocess text data."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())

    def load_and_preprocess_data(self, file_path, sample_size=160000):
        print("Loading Sentiment140 dataset...")
        columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(file_path, encoding='latin-1', names=columns)

        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        print(f"Dataset shape: {df.shape}")
        df['target'] = df['target'].map({0: 0, 4: 1})
        df['cleaned_text'] = df['text'].astype(str).apply(self.clean_text)
        df = df[df['cleaned_text'].str.strip().astype(bool)]
        return df[['cleaned_text', 'target']]

    def create_features(self, df, fit_vectorizer=True):
        print("Creating TF-IDF features...")
        if fit_vectorizer:
            X = self.vectorizer.fit_transform(df['cleaned_text'])
            with open('tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
        else:
            X = self.vectorizer.transform(df['cleaned_text'])

        y = df['target'].values
        if self.convert_to_dense:
            print("Converting sparse matrix to dense (float32)...")
            X = X.astype(np.float32).toarray()

        return X, y

    def create_federated_splits(self, X, y, num_clients=5, alpha=0.5):
        print(f"Creating federated splits for {num_clients} clients...")
        num_classes = len(np.unique(y))
        sorted_indices = np.argsort(y)
        client_data_indices = []

        for client in range(num_clients):
            proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
            client_indices = []

            for class_idx in range(num_classes):
                class_indices = sorted_indices[y[sorted_indices] == class_idx]
                n_class_samples = int(proportions[class_idx] * len(class_indices) / num_clients)
                start = client * n_class_samples
                end = start + n_class_samples
                if end <= len(class_indices):
                    client_indices.extend(class_indices[start:end])

            client_data_indices.append(client_indices)

        client_datasets = []
        for indices in client_data_indices:
            if len(indices) > 0:
                client_X = X[indices]
                client_y = y[indices]
                X_train, X_test, y_train, y_test = train_test_split(
                    client_X, client_y, test_size=0.2, random_state=42, stratify=client_y)
                client_datasets.append({
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test
                })

        return client_datasets


def prepare_federated_data(dataset_path, convert_to_dense=False, max_features=3000):
    """Main function to prepare federated data."""
    preprocessor = SentimentDataPreprocessor(
        max_features=max_features,
        convert_to_dense=convert_to_dense
    )
    df = preprocessor.load_and_preprocess_data(dataset_path)
    X, y = preprocessor.create_features(df)
    client_datasets = preprocessor.create_federated_splits(X, y, num_clients=5)

    os.makedirs('federated_data', exist_ok=True)
    for i, dataset in enumerate(client_datasets):
        with open(f'federated_data/client_{i}_data.pkl', 'wb') as f:
            pickle.dump(dataset, f)

    print(f"Created {len(client_datasets)} client datasets:")
    for i, dataset in enumerate(client_datasets):
        print(f" - Client {i}: Train samples: {len(dataset['y_train'])}, Test samples: {len(dataset['y_test'])}")

    return client_datasets


if __name__ == "__main__":
    dataset_path = "try3/sentiment140.csv"  # Replace with actual path
    # Use convert_to_dense=True only if necessary
    prepare_federated_data(dataset_path, convert_to_dense=False, max_features=3000)