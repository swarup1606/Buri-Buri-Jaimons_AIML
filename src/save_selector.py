import pickle
import os
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import numpy as np

def load_data_and_model():
    """
    Load the preprocessed data and trained model
    """
    # Load preprocessed data
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    
    # Load model
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/xgboost_basic_improved.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return X_train, y_train, model

def create_and_save_selector(X_train, y_train, model):
    """
    Create a feature selector based on the model's feature importances
    and save it
    """
    # Create selector
    selector = SelectFromModel(model, threshold='median')
    selector.fit(X_train, y_train)
    
    # Save selector
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/selector_basic.pkl', 'wb') as f:
        pickle.dump(selector, f)
    
    print(f"Selector saved. Selected {sum(selector.get_support())} features out of {len(selector.get_support())}")

def main():
    # Create directory if it doesn't exist
    os.makedirs('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models', exist_ok=True)
    
    try:
        # Try to load data and model
        X_train, y_train, model = load_data_and_model()
        create_and_save_selector(X_train, y_train, model)
    except FileNotFoundError:
        print("Error: Required files not found.")
        print("Please run model_training.py first to generate the necessary files.")

if __name__ == "__main__":
    main()