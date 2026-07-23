import os
import sys
import pandas as pd

# Add the src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.explain_gnn import explain_transaction

def main():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "fraud_dataset_cleaned.csv"))
    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found at {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    if df.empty:
        print("Error: clean dataset is empty.")
        return
        
    # Get a sample transaction
    sample_txn = df.iloc[0]
    txn_id = sample_txn['transaction_id']
    is_fraud = sample_txn['is_fraud']
    print(f"Selected Transaction: {txn_id} (Actual Label: {'Fraud' if is_fraud == 1 else 'Legitimate'})")
    
    print("\nRunning GNNExplainer...")
    try:
        feature_importances, subgraph_edges = explain_transaction(txn_id, epochs=20)
        
        print("\nTop 5 Driving Features:")
        for i, item in enumerate(feature_importances[:5]):
            print(f"{i+1}. {item['feature']}: {item['importance']:.4f}")
            
        print("\nLocal Subgraph Neighbor Edges:")
        for i, edge in enumerate(subgraph_edges[:10]):
            print(f"{i+1}. {edge['source']} <--> {edge['target']} (Importance: {edge['importance']:.4f})")
            
        print("\nSUCCESS: GNNExplainer run completed successfully!")
    except Exception as e:
        print(f"\nError running explainability: {e}")

if __name__ == "__main__":
    main()
