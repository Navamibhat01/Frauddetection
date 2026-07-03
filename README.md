# Fraud Detection using Graph Neural Networks (GNN)

A machine learning system for detecting fraudulent transactions using Graph Neural Networks and anomaly detection techniques.

## Project Overview

This project implements an advanced fraud detection system that combines:
- **Graph Neural Networks (GNN)** for modeling transaction relationships and patterns
- **Anomaly Detection** algorithms for identifying suspicious transactions
- **Web-based Dashboard** for visualization and risk assessment
- **Data Processing Pipeline** for cleaning and preparing raw transaction data

## Features

- 📊 Transaction graph analysis and visualization
- 🤖 GNN-based fraud prediction model
- 🔍 Anomaly detection with statistical analysis
- 🌐 Interactive web interface for risk assessment
- 📈 Comprehensive analytics and reporting
- 🎯 High-risk transaction identification

## Project Structure

```
fraud-detection-gnn/
├── src/                          # Main source code
│   ├── app.py                    # Flask/main application
│   ├── anomaly_detection/        # Anomaly detection module
│   │   └── anomaly_detector.py   # Anomaly detection algorithms
│   └── graph/                    # Graph and GNN models
│       ├── f_graph.py            # Fraud graph construction
│       ├── train_gnn.py          # GNN model training
│       ├── pickel.py             # Data serialization
│       └── models/               # Pre-trained models
├── data/                         # Data directory
│   ├── raw/                      # Raw transaction data
│   ├── processed/                # Cleaned data
│   ├── output/                   # Analysis results
│   └── graph/                    # Graph embeddings and labels
├── web/                          # Web interface
│   ├── index.html                # Dashboard HTML
│   ├── script.js                 # Frontend logic
│   └── styles.css                # Styling
├── visualizations/               # Visualization scripts
│   ├── accuracy_graph.py         # Model accuracy plots
│   └── graph_visualization.html  # Network visualization
├── lib/                          # JavaScript libraries
│   ├── tom-select/               # Select component
│   └── vis-9.1.2/                # Network visualization
├── logs/                         # Application logs
├── run_web_app.py                # Web server entry point
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Node.js (optional, for frontend development)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd fraud-detection-gnn
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**
   - Place raw transaction data in `data/raw/fraud_dataset.csv`
   - Run preprocessing: `python src/preprocess/clean_dataset.py`

## Usage

### Running the Web Application

```bash
python run_web_app.py
```

The application will start at `http://localhost:5000` (or as configured)

### Training the GNN Model

```bash
python src/graph/train_gnn.py
```

### Anomaly Detection

```bash
python src/anomaly_detection/anomaly_detector.py
```

### Data Processing

```bash
python src/preprocess/clean_dataset.py
```

### Visualizations

Generate accuracy graphs:
```bash
python visualizations/accuracy_graph.py
```

View transaction network:
Open `visualizations/graph_visualization.html` in a browser

## Key Components

### Graph Module (`src/graph/`)
- **f_graph.py**: Constructs transaction graphs and relationships
- **train_gnn.py**: Trains the GNN model on graph data
- **pickel.py**: Handles model persistence and loading

### Anomaly Detection (`src/anomaly_detection/`)
- Detects unusual transaction patterns
- Uses statistical methods and ML algorithms
- Generates anomaly scores

### Web Interface (`web/`)
- Interactive dashboard for viewing transactions
- Risk scoring visualization
- Transaction filtering and search
- Real-time analysis results

## Data Files

- `fraud_dataset.csv` - Raw transaction data
- `fraud_dataset_cleaned.csv` - Processed dataset
- `fraud_risk_scores.csv` - Model predictions
- `high_risk_transactions.csv` - Flagged transactions
- Node/transaction embeddings stored in `data/graph/`

## Output

Results are saved in `data/output/`:
- `fraud_risk_scores.csv` - Fraud probability scores
- `high_risk_transactions.csv` - Transactions flagged as suspicious
- `fraud_risk_results.csv` - Detailed analysis results

## Results

The system generates comprehensive reports:
- `final_blocked_report.csv` - Blocked transactions
- `final_thesis_submission.csv` - Research results
- `final_tuned_results.csv` - Optimized model performance

## Technologies Used

- **Machine Learning**: PyTorch, Scikit-learn, NumPy, Pandas
- **Graph Analysis**: DGL/PyG (Graph Neural Networks)
- **Web Framework**: Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Vis.js, Plotly
- **Data Processing**: Pandas, NumPy

## Performance

Model evaluation metrics stored in:
- Accuracy and precision reports
- ROC/AUC curves
- Confusion matrices
- F1 scores

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and commit: `git commit -am 'Add feature'`
3. Push to the branch: `git push origin feature/your-feature`
4. Submit a pull request

## License

[Add your license here]

## Contact

[Add contact information]

## Acknowledgments

- Transaction dataset source: [Credit fraud detection dataset]
- References: [Academic papers on GNN fraud detection]

---

For more information, check the logs in `logs/` directory and review configuration in the respective module files.
