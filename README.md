# Federated ML for Twitter Sentiment Analysis

A privacy-preserving machine learning system that performs sentiment analysis on Twitter data using federated learning principles. This project enables multiple parties to collaboratively train sentiment analysis models without sharing raw tweet data.

## Overview

This project implements a federated learning framework for Twitter sentiment analysis, allowing distributed training across multiple data sources while maintaining data privacy and security. The system uses natural language processing techniques to classify tweet sentiment (positive, negative, neutral) without centralizing sensitive social media data.

## Features

- **Privacy-Preserving**: Raw tweet data never leaves local nodes
- **Distributed Training**: Multiple clients can participate in model training
- **Real-time Analysis**: Process live Twitter streams for sentiment classification
- **Scalable Architecture**: Supports horizontal scaling across multiple nodes
- **Model Aggregation**: Secure aggregation of model updates using federated averaging
- **Performance Monitoring**: Track model performance across federated rounds

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Node 1 │    │   Client Node 2 │    │   Client Node N │
│                 │    │                 │    │                 │
│ Local Twitter   │    │ Local Twitter   │    │ Local Twitter   │
│ Data + Model    │    │ Data + Model    │    │ Data + Model    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │ Model Updates        │ Model Updates        │ Model Updates
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Federated Server       │
                    │                          │
                    │ - Model Aggregation      │
                    │ - Global Model Update    │
                    │ - Round Coordination     │
                    └──────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager


### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/federated-twitter-sentiment.git
cd federated-twitter-sentiment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets from Kaggle:
```bash
# Setup Kaggle API credentials first (see Dataset section)
bash scripts/download_data.sh
```

4. Prepare federated data splits:
```bash
python scripts/data_preparation.py --num_clients 5 --data_path data/raw/
```

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

### Environment Variables

```env
# Kaggle API Configuration  
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Federated Learning Configuration
FL_ROUNDS=10
MIN_CLIENTS=3
CLIENT_FRACTION=0.8
LEARNING_RATE=0.01

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=8080

# Data Configuration
DATASET_PATH=data/processed/
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1
TEST_SPLIT=0.1
```

### Model Configuration

```yaml
# config/model_config.yaml
model:
  type: "lstm"
  embedding_dim: 128
  hidden_dim: 64
  num_layers: 2
  dropout: 0.3
  
training:
  batch_size: 32
  local_epochs: 5
  optimizer: "adam"
  loss_function: "categorical_crossentropy"

preprocessing:
  max_sequence_length: 100
  vocab_size: 10000
  remove_stopwords: true
  lowercase: true
```

## Usage

### Running the Federated Server

```bash
python federated_server.py --config config/server_config.yaml
```

### Running Client Nodes

```bash
# Terminal 1 - Client 1
python federated_client.py --client_id 1 --data_path data/processed/client1_tweets.csv

# Terminal 2 - Client 2  
python federated_client.py --client_id 2 --data_path data/processed/client2_tweets.csv

# Terminal 3 - Client 3
python federated_client.py --client_id 3 --data_path data/processed/client3_tweets.csv
```

### Real-time Sentiment Analysis

```bash
# Analyze sentiment using trained federated model
python sentiment_analyzer.py --model_path models/global_model.pkl --input "This movie is amazing!"

# Batch analysis from file  
python sentiment_analyzer.py --model_path models/global_model.pkl --batch_file data/test_tweets.csv
```

## Dataset

### Data Source
This project uses Twitter sentiment analysis datasets from **Kaggle**:

- **Primary Dataset**: [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (Sentiment140)
- **Alternative Dataset**: [Twitter Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction)
- **Supplementary**: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

### Data Download Instructions

1. **Install Kaggle API**:
```bash
pip install kaggle
```

2. **Setup Kaggle credentials**:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Create new API token (downloads kaggle.json)
   - Place in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

3. **Download datasets**:
```bash
# Download Sentiment140 dataset
kaggle datasets download -d kazanova/sentiment140 -p data/raw/

# Download Twitter Airline Sentiment
kaggle datasets download -d crowdflower/twitter-airline-sentiment -p data/raw/

# Extract files
cd data/raw && unzip "*.zip"
```

### Data Format

The system expects tweet data in CSV format with the following columns:

```csv
tweet_id,text,sentiment,timestamp,user_id
1234567890,"Great day today! #happy",positive,2024-01-15 10:30:00,user123
1234567891,"Traffic is terrible...",negative,2024-01-15 10:31:00,user456
1234567892,"Meeting went well",positive,2024-01-15 10:32:00,user789
```

### Sentiment Labels
- `positive` (or `4` in Sentiment140): Positive sentiment
- `negative` (or `0` in Sentiment140): Negative sentiment  
- `neutral` (or `2` in Sentiment140): Neutral sentiment

### Dataset Statistics
| Dataset | Size | Positive | Negative | Neutral |
|---------|------|----------|----------|---------|
| Sentiment140 | 1.6M tweets | 800K | 800K | - |
| Airline Sentiment | 14.6K tweets | 2.4K | 9.2K | 3.1K |
| Combined | ~1.61M tweets | ~802K | ~809K | ~3.1K |

### Related Resources

- **Kaggle Code Examples**: Explore community notebooks and code examples for the Sentiment140 dataset:
  - [Data Exploration & Visualization](https://www.kaggle.com/datasets/kazanova/sentiment140/code)
  - Popular approaches: LSTM, BERT, Traditional ML methods
  - Preprocessing techniques and feature engineering examples
- **Academic Paper**: [Twitter Sentiment Classification using Distant Supervision](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf) - Original Sentiment140 research
- **Federated Learning Resources**:
  - [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
  - [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)



## Model Performance

### Baseline Metrics

| Metric | Centralized | Federated (3 clients) | Federated (5 clients) |
|--------|-------------|----------------------|----------------------|
| Accuracy | 0.892 | 0.876 | 0.883 |
| Precision | 0.885 | 0.871 | 0.879 |
| Recall | 0.890 | 0.875 | 0.881 |
| F1-Score | 0.887 | 0.873 | 0.880 |

![federated_learning_results](https://github.com/user-attachments/assets/70c55da8-ba09-446f-ad81-6a707198bba3)


### Privacy Analysis

- **Data Localization**: Raw tweets never leave client devices
- **Differential Privacy**: Optional noise addition to model updates
- **Secure Aggregation**: Encrypted model parameter sharing
- **Communication Overhead**: ~95% reduction vs. centralized approach

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Future Enhancements

### Planned Features
- Docker containerization for easy deployment
- Kubernetes orchestration for scalable federated training
- Web dashboard for monitoring federated rounds
- Support for additional NLP models (BERT, RoBERTa)
- Differential privacy implementation
- Cross-platform mobile client support

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure server is running and firewall allows connections
2. **Model Convergence Issues**: Adjust learning rate or increase local epochs
3. **Memory Errors**: Reduce batch size or model complexity
4. **Twitter API Limits**: Implement rate limiting and retry logic

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Monitor federated rounds:
```bash
tail -f logs/federated_server.log
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Datasets**: Twitter sentiment datasets from [Kaggle](https://www.kaggle.com/)
  - [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) by Stanford University
  - [Twitter Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) by Crowdflower
- [Flower Framework](https://flower.dev/) for federated learning infrastructure
- [Transformers](https://huggingface.co/transformers/) for NLP models
- [Scikit-learn](https://scikit-learn.org/) for machine learning utilities



