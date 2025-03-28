import pickle
import os
import numpy as np
from sklearn.feature_selection import SelectFromModel

def create_selector():
    # Create models directory if it doesn't exist
    os.makedirs('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models', exist_ok=True)
    
    # Load model
    model_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/xgboost_basic_improved.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Create a selector based on the model's feature importances
        selector = SelectFromModel(model, prefit=True)
        
        # Save the selector
        selector_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/selector_basic.pkl'
        with open(selector_path, 'wb') as f:
            pickle.dump(selector, f)
        
        print(f"Selector created and saved to {selector_path}")
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    except Exception as e:
        print(f"Error creating selector: {str(e)}")

if __name__ == "__main__":
    create_selector()