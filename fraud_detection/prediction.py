import pandas as pd
import pickle
import os
import sys

def load_model(model_path):
    """Load a saved model from the specified path."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_data(input_file):
    """Load and preprocess the input data."""
    # Load data
    df = pd.read_csv(input_file)
    
    # Retain Time and Amount columns for output
    time_amount = df[['Time', 'Amount']] if 'Time' in df.columns and 'Amount' in df.columns else None
    
    # Drop non-feature columns
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])
    
    return df, time_amount

def predict_fraudulent_transactions(input_file, model, output_file):
    """Predict fraudulent transactions and save the results."""
    # Preprocess the input data
    features, time_amount = preprocess_data(input_file)
    
    # Predict fraudulent transactions
    predictions = model.predict(features)
    
    # Combine predictions with Time and Amount
    results = time_amount.copy()
    results['Prediction'] = predictions
    
    # Filter only fraudulent transactions
    fraudulent_transactions = results[results['Prediction'] == 1]
    
    # Save fraudulent transactions to a CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fraudulent_transactions.to_csv(output_file, index=False)
    print(f"Fraudulent transactions saved to '{output_file}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prediction.py <input_file.csv>")
        sys.exit(1)
    
    input_file = os.path.join("fraud_detection/data",sys.argv[1])
    output_file = "fraud_detection/output"
    
    # Load the model
    model_path = "fraud_detection/models/fraud_detection_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model file not found at '{model_path}'.")
        sys.exit(1)
    
    model = load_model(model_path)
    
    # Run predictions
    predict_fraudulent_transactions(input_file, model, output_file)
