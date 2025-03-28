import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def load_data(file_path='c:/Users/Vaibhav/OneDrive/Desktop/predict_job/data/hiring_data.csv'):
    """
    Load historical hiring data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)

def preprocess_data(data, text_features=False):
    """
    Preprocess the hiring data:
    - Handle missing values
    - Convert categorical variables
    - Scale numerical features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw hiring data
    text_features : bool
        Whether to include text feature processing
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, preprocessor
    """
    # Fill missing values
    data = data.copy()
    data['certifications'] = data['certifications'].fillna('None')
    
    # Separate features and target
    if 'hired' in data.columns:
        y = data['hired']
        X = data.drop(['hired', 'performance_score'], axis=1, errors='ignore')
    elif 'performance_score' in data.columns:
        y = data['performance_score']
        X = data.drop(['performance_score'], axis=1, errors='ignore')
    else:
        raise ValueError("No target variable found in the dataset")
    
    # Identify numerical and categorical columns
    numerical_cols = ['years_experience', 'skill_match_score']
    categorical_cols = ['education', 'job_level', 'industry']
    text_cols = ['skills', 'past_job_titles', 'certifications', 'required_skills', 'job_title']
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create preprocessor
    if text_features:
        # Include text feature processing
        text_transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=50))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols),
                ('text_skills', text_transformer, 'skills'),
                ('text_job_title', text_transformer, 'job_title'),
                ('text_required_skills', text_transformer, 'required_skills')
            ])
    else:
        # Basic preprocessing without text features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def get_feature_names(preprocessor, X):
    """
    Get feature names after preprocessing
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Fitted preprocessor
    X : pandas.DataFrame
        Original feature dataframe
        
    Returns:
    --------
    list
        Feature names after preprocessing
    """
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            # Get one-hot encoded feature names
            encoder = transformer.named_steps['onehot']
            categories = encoder.categories_
            for i, category in enumerate(categories):
                col = columns[i]
                for cat_val in category:
                    feature_names.append(f"{col}_{cat_val}")
        elif name.startswith('text_'):
            # Get TF-IDF feature names
            vectorizer = transformer.named_steps['tfidf']
            for feature in vectorizer.get_feature_names_out():
                feature_names.append(f"{name}_{feature}")
    
    return feature_names

if __name__ == "__main__":
    # Test data loading and preprocessing
    try:
        data = load_data()
        print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Basic preprocessing
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)
        print(f"Preprocessed data shapes: X_train {X_train.shape}, X_test {X_test.shape}")
        
        # Advanced preprocessing with text features
        X_train_text, X_test_text, y_train_text, y_test_text, preprocessor_text = preprocess_data(data, text_features=True)
        print(f"Preprocessed data with text features shapes: X_train {X_train_text.shape}, X_test {X_test_text.shape}")
        
    except FileNotFoundError:
        print("Data file not found. Please run generate_data.py first to create sample data.")