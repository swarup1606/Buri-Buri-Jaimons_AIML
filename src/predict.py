import pickle
import pandas as pd
import numpy as np
import os

def load_model(model_path, preprocessor_path, selector_path=None):
    """
    Load the trained model and preprocessor
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    preprocessor_path : str
        Path to the saved preprocessor
    selector_path : str, optional
        Path to the saved feature selector
        
    Returns:
    --------
    tuple
        (model, preprocessor, selector)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    selector = None
    if selector_path and os.path.exists(selector_path):
        with open(selector_path, 'rb') as f:
            selector = pickle.load(f)
    
    return model, preprocessor, selector

def predict_candidate(candidate_data, model, preprocessor, selector=None):
    """
    Predict whether a candidate is a good fit for a job
    
    Parameters:
    -----------
    candidate_data : dict
        Dictionary containing candidate information
    model : object
        Trained model
    preprocessor : object
        Data preprocessor
    selector : object, optional
        Feature selector
        
    Returns:
    --------
    dict
        Prediction results
    """
    # Convert candidate data to DataFrame
    candidate_df = pd.DataFrame([candidate_data])
    
    # Preprocess the data
    X = preprocessor.transform(candidate_df)
    
    # Apply feature selection if provided
    if selector is not None:
        X = selector.transform(X)
    else:
        # If no selector is provided but there's a feature mismatch,
        # try to infer the expected number of features from the model
        try:
            expected_features = model.n_features_in_
            if X.shape[1] != expected_features:
                print(f"Warning: Feature shape mismatch. Model expects {expected_features} features, got {X.shape[1]}.")
                print("Attempting to use model's feature_importances_ to select features...")
                
                # If model has feature_importances_, use them to select top features
                if hasattr(model, 'feature_importances_'):
                    # Get indices of top features
                    top_indices = np.argsort(model.feature_importances_)[-expected_features:]
                    X = X[:, top_indices]
                    print(f"Selected {X.shape[1]} features based on importance.")
        except Exception as e:
            print(f"Error during feature selection: {str(e)}")
    
    # Make prediction
    try:
        prediction_proba = model.predict_proba(X)[0, 1]
        prediction = model.predict(X)[0]
    except ValueError as e:
        if "Feature shape mismatch" in str(e):
            print(f"Error: {str(e)}")
            print("Please ensure the model and feature selector are compatible.")
            return {
                'prediction': None,
                'probability': 0.0,
                'recommendation': 'Error: Feature mismatch'
            }
        else:
            raise
    
    # Return results
    return {
        'prediction': bool(prediction),
        'probability': float(prediction_proba),
        'recommendation': 'Recommended' if prediction == 1 else 'Not Recommended'
    }

def inspect_preprocessor(preprocessor):
    """
    Inspect the preprocessor to understand what columns it expects
    
    Parameters:
    -----------
    preprocessor : object
        Data preprocessor
        
    Returns:
    --------
    list
        List of expected column names
    """
    # Try to extract column names from the preprocessor
    try:
        # For ColumnTransformer
        if hasattr(preprocessor, 'transformers_'):
            all_columns = []
            for name, transformer, columns in preprocessor.transformers_:
                if columns is not None:
                    all_columns.extend(columns)
            return all_columns
        # For Pipeline with ColumnTransformer as first step
        elif hasattr(preprocessor, 'steps') and hasattr(preprocessor.steps[0][1], 'transformers_'):
            column_transformer = preprocessor.steps[0][1]
            all_columns = []
            for name, transformer, columns in column_transformer.transformers_:
                if columns is not None:
                    all_columns.extend(columns)
            return all_columns
    except Exception as e:
        print(f"Error inspecting preprocessor: {str(e)}")
    
    return []

def main():
    # Load model, preprocessor, and selector
    model_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/xgboost_basic_improved.pkl'
    preprocessor_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/preprocessor_basic.pkl'
    selector_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/selector_basic.pkl'
    
    model, preprocessor, selector = load_model(model_path, preprocessor_path, selector_path)
    
    # Inspect preprocessor to see what columns it expects
    expected_columns = inspect_preprocessor(preprocessor)
    print("\nPreprocessor expects these columns:")
    print(expected_columns)
    
    # Example candidate data with the correct column names based on the expected columns
    candidate = {
        'years_experience': 3,
        'skill_match_score': 0.75,
        'education': 'Bachelor',
        'job_level': 'Mid-level',
        'industry': 'Technology'
    }
    
    # Make prediction
    result = predict_candidate(candidate, model, preprocessor, selector)
    
    # Print results
    print("\nCandidate Prediction:")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['probability']:.2%}")

if __name__ == "__main__":
    main()