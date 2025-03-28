import pandas as pd
import numpy as np
import pickle
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_model_and_data():
    """
    Load the trained model, preprocessor, and test data
    """
    model_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/xgboost_basic_improved.pkl'
    preprocessor_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/preprocessor_basic.pkl'
    selector_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/selector_basic.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(selector_path, 'rb') as f:
        selector = pickle.load(f)
    
    # Load test data
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    return model, preprocessor, selector, X_test, y_test

def evaluate_model(model, X, y, selector=None):
    """
    Evaluate model performance
    """
    # Apply feature selection if provided
    if selector is not None:
        X = selector.transform(X)
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    return metrics

def log_metrics(metrics, log_file='c:/Users/Vaibhav/OneDrive/Desktop/predict_job/logs/model_metrics.csv'):
    """
    Log model metrics to a CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Add timestamp
    metrics['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Check if log file exists
    if os.path.exists(log_file):
        # Append to existing log
        existing_log = pd.read_csv(log_file)
        updated_log = pd.concat([existing_log, metrics_df], ignore_index=True)
        updated_log.to_csv(log_file, index=False)
    else:
        # Create new log
        metrics_df.to_csv(log_file, index=False)
    
    print(f"Metrics logged to {log_file}")

def plot_metrics_trend(log_file='c:/Users/Vaibhav/OneDrive/Desktop/predict_job/logs/model_metrics.csv'):
    """
    Plot trend of model metrics over time
    """
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found")
        return
    
    # Load metrics log
    metrics_log = pd.read_csv(log_file)
    
    # Convert timestamp to datetime
    metrics_log['timestamp'] = pd.to_datetime(metrics_log['timestamp'])
    
    # Plot metrics trend
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    for metric in metrics_to_plot:
        plt.plot(metrics_log['timestamp'], metrics_log[metric], marker='o', label=metric)
    
    plt.title('Model Performance Metrics Over Time')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/logs/metrics_trend.png')
    plt.close()
    
    print("Metrics trend plot saved to logs/metrics_trend.png")

def main():
    # Load model and data
    print("Loading model and data...")
    model, preprocessor, selector, X_test, y_test = load_model_and_data()
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, selector)
    
    # Print metrics
    print("\nModel Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Log metrics
    print("\nLogging metrics...")
    log_metrics(metrics)
    
    # Plot metrics trend
    print("\nPlotting metrics trend...")
    plot_metrics_trend()
    
    print("\nDone!")

if __name__ == "__main__":
    main()