import os
import sys
from data_preprocessing import prepare_federated_data
from federated_model import run_federated_learning, plot_results


def main():
    """Main execution function."""

    print("=== Federated Learning for Sentiment Analysis ===\n")

    # Step 1: Check if data is preprocessed
    if not os.path.exists('federated_data'):
        print("Preprocessed data not found. Please run data preprocessing first.")
        print("1. Download the Sentiment140 dataset from Kaggle")
        print("2. Place it as 'sentiment140.csv' in the current directory")
        print("3. Run the preprocessing script")

        dataset_path = input("Enter path to sentiment140.csv (or press Enter for 'sentiment140.csv'): ").strip()
        if not dataset_path:
            dataset_path = "sentiment140.csv"

        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            return

        print("Preprocessing data...")
        prepare_federated_data(dataset_path)

    # Step 2: Run federated learning
    print("\nStarting federated learning...")
    server, clients, results = run_federated_learning(
        num_rounds=15,
        num_clients=5,
        epochs_per_round=3
    )

    # Step 3: Visualize results
    print("\nGenerating visualizations...")
    plot_results(results)

    # Step 4: Detailed evaluation
    # detailed_evaluation(server, clients)

    print("\n=== Federated Learning Complete ===")
    print("Results saved:")
    print("- Model: federated_sentiment_model.pth")
    print("- Plots: federated_learning_results.png, confusion_matrix.png")
    print("- Vectorizer: tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()