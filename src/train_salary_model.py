import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Create models directory if it doesn't exist
os.makedirs('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models', exist_ok=True)
os.makedirs('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/plots', exist_ok=True)

def load_and_preprocess_data():
    """
    Load and preprocess the salary dataset
    """
    # Load data
    data_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/data/Salary_data.csv'
    df = pd.read_csv(data_path)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Identify target column (assuming it's named 'salary' or it's the last column)
    target_col = 'salary' if 'salary' in df.columns else df.columns[-1]
    
    # Define features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split data into train, validation, and test sets (70%, 15%, 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df.columns

def create_preprocessor(X_train):
    """
    Create a preprocessing pipeline for numerical and categorical features
    """
    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def train_and_evaluate_models(X_train, X_val, y_train, y_val, preprocessor):
    """
    Train different regression models and evaluate their performance
    """
    # Create pipelines with preprocessing and different models
    models = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(random_state=42))
        ]),
        'LightGBM': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(random_state=42))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        results[name] = {
            'model': model,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    return results

def optimize_best_model(best_model_name, best_model, X_train, X_val, y_train, y_val):
    """
    Optimize hyperparameters for the best performing model
    """
    print(f"\nOptimizing hyperparameters for {best_model_name}...")
    
    # Define hyperparameter grid based on the model type
    param_grid = {}
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7],
            'regressor__colsample_bytree': [0.7, 0.8, 0.9]
        }
    elif best_model_name == 'LightGBM':
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7],
            'regressor__num_leaves': [31, 50, 70]
        }
    
    # If we have a parameter grid for the selected model
    if param_grid:
        # Create grid search
        grid_search = GridSearchCV(
            best_model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        optimized_model = grid_search.best_estimator_
        
        # Evaluate optimized model
        y_pred = optimized_model.predict(X_val)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return optimized_model, grid_search.best_params_
    
    return best_model, {}

def evaluate_final_model(model, X_test, y_test):
    """
    Evaluate the final model on the test set
    """
    print("\nEvaluating final model on test set...")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title('Actual vs Predicted Salary')
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/plots/actual_vs_predicted.png')
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
    plt.xlabel('Predicted Salary')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/plots/residuals.png')
    plt.close()
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def save_model(model, model_name):
    """
    Save the trained model to disk
    """
    model_path = f'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/salary_{model_name.lower().replace(" ", "_")}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    return model_path

def main():
    print("Starting salary prediction model training...")
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, columns = load_and_preprocess_data()
    
    # Create preprocessor
    preprocessor = create_preprocessor(X_train)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_val, y_train, y_val, preprocessor)
    
    # Find best model based on R² score
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with R² Score: {results[best_model_name]['r2']:.4f}")
    
    # Optimize best model
    optimized_model, best_params = optimize_best_model(best_model_name, best_model, X_train, X_val, y_train, y_val)
    
    # Evaluate final model on test set
    final_metrics = evaluate_final_model(optimized_model, X_test, y_test)
    
    # Save model
    model_path = save_model(optimized_model, best_model_name)
    
    print("\nSalary prediction model training completed!")
    return model_path, final_metrics

if __name__ == "__main__":
    main()