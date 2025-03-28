import pickle
import pandas as pd
import numpy as np
import os

def load_salary_model():
    """
    Load the trained salary prediction model
    """
    try:
        # Find the most recent salary model
        models_dir = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models'
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        salary_models = [f for f in os.listdir(models_dir) if f.startswith('salary_') and f.endswith('.pkl')]
        
        if not salary_models:
            # If no model exists, return a simple linear model
            print("No salary prediction model found. Using a simple default model.")
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            return model
        
        # Sort by modification time (newest first)
        salary_models.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
        
        # Load the most recent model
        model_path = os.path.join(models_dir, salary_models[0])
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Loaded salary model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading salary model: {str(e)}")
        # Return a simple model if there's an error
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        return model

def predict_salary(candidate_data):
    """
    Predict salary based on candidate data
    
    Args:
        candidate_data (dict): Dictionary containing candidate information
        
    Returns:
        float: Predicted salary
    """
    try:
        # Load model
        model = load_salary_model()
        
        # Convert to DataFrame
        candidate_df = pd.DataFrame([candidate_data])
        
        # If it's a simple linear model (fallback), train it on some basic data
        if isinstance(model, type(pd.DataFrame()).__mro__[-2]):  # Check if it's a basic model
            # Generate some simple training data
            X_train = pd.DataFrame({
                'years_experience': [0, 5, 10, 15, 20, 25],
                'education_val': [0, 1, 2, 3, 4, 4],  # Numeric values for education levels
                'job_level_val': [0, 1, 2, 3, 3, 3],  # Numeric values for job levels
                'industry_val': [0, 1, 2, 3, 4, 5],   # Numeric values for industries
                'location_val': [0, 1, 2, 0, 1, 2]    # Numeric values for locations
            })
            
            # Generate target values (simple formula)
            y_train = 30000 + X_train['years_experience'] * 2000 + X_train['education_val'] * 10000 + \
                     X_train['job_level_val'] * 20000 + X_train['industry_val'] * 5000 + X_train['location_val'] * 5000
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Convert categorical features to numeric for the candidate
            education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
            job_level_map = {'Entry-level': 0, 'Mid-level': 1, 'Senior': 2, 'Executive': 3}
            industry_map = {'Technology': 0, 'Finance': 1, 'Healthcare': 2, 'Education': 3, 'Manufacturing': 4, 'Retail': 5}
            location_map = {'Urban': 0, 'Suburban': 1, 'Rural': 2}
            
            # Create features for prediction
            X_pred = pd.DataFrame({
                'years_experience': [candidate_data['years_experience']],
                'education_val': [education_map.get(candidate_data['education'], 2)],  # Default to Bachelor
                'job_level_val': [job_level_map.get(candidate_data['job_level'], 1)],  # Default to Mid-level
                'industry_val': [industry_map.get(candidate_data['industry'], 0)],     # Default to Technology
                'location_val': [location_map.get(candidate_data['location'], 0)]      # Default to Urban
            })
            
            # Make prediction
            predicted_salary = model.predict(X_pred)[0]
        else:
            # Make prediction using the loaded model
            # The model includes preprocessing steps, so we can directly use it
            predicted_salary = model.predict(candidate_df)[0]
        
        return predicted_salary
    
    except Exception as e:
        print(f"Error predicting salary: {str(e)}")
        
        # Fallback to a simple calculation if prediction fails
        years_exp = candidate_data.get('years_experience', 0)
        
        # Basic salary calculation
        base_salary = 30000
        exp_factor = years_exp * 2000
        
        # Education factor
        edu_mapping = {
            'High School': 0,
            'Associate': 5000,
            'Bachelor': 15000,
            'Master': 25000,
            'PhD': 35000
        }
        edu_factor = edu_mapping.get(candidate_data.get('education', 'Bachelor'), 15000)
        
        # Job level factor
        level_mapping = {
            'Entry-level': 0,
            'Mid-level': 20000,
            'Senior': 40000,
            'Executive': 80000
        }
        level_factor = level_mapping.get(candidate_data.get('job_level', 'Mid-level'), 20000)
        
        # Industry factor
        industry_mapping = {
            'Technology': 15000,
            'Finance': 12000,
            'Healthcare': 10000,
            'Education': 5000,
            'Manufacturing': 8000,
            'Retail': 3000
        }
        industry_factor = industry_mapping.get(candidate_data.get('industry', 'Technology'), 15000)
        
        # Location factor
        location_mapping = {
            'Urban': 10000,
            'Suburban': 5000,
            'Rural': 0
        }
        location_factor = location_mapping.get(candidate_data.get('location', 'Urban'), 10000)
        
        # Calculate salary
        predicted_salary = base_salary + exp_factor + edu_factor + level_factor + industry_factor + location_factor
        
        return predicted_salary