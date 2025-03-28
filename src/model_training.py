import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Try to import LIME instead of SHAP
try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("LIME library not available. Install with 'pip install lime' for feature importance visualization.")
    LIME_AVAILABLE = False

from data_preparation import load_data, preprocess_data

# Add these imports for better hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.feature_selection import SelectFromModel

def train_xgboost_model(X_train, y_train, X_test, y_test, use_grid_search=True):
    """
    Train an XGBoost model for candidate-job fit prediction
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
        
    Returns:
    --------
    tuple
        (model, metrics)
    """
    if use_grid_search:
        # Define more comprehensive parameter grid for better tuning
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'scale_pos_weight': [1, 3, 5]  # Helps with class imbalance
        }
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        
        # Use RandomizedSearchCV instead of GridSearchCV for efficiency
        grid_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter settings sampled
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        # Create and train XGBoost model with default parameters
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42
        )
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return model, metrics

# Add a new function for feature selection
def select_important_features(X_train, y_train, X_test, threshold='median'):
    """
    Select important features using XGBoost's feature importance
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Test features
    threshold : str or float
        Threshold for feature selection
        
    Returns:
    --------
    tuple
        (X_train_selected, X_test_selected, selector)
    """
    # Train a model for feature selection
    selector_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    
    # Create feature selector
    selector = SelectFromModel(
        estimator=selector_model,
        threshold=threshold
    )
    
    # Fit and transform
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")
    
    return X_train_selected, X_test_selected, selector

def analyze_feature_importance(model, feature_names=None):
    """
    Analyze and visualize feature importance
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    feature_names : list
        Names of features
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for visualization
    if feature_names is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
    else:
        importance_df = pd.DataFrame({
            'Feature': [f'Feature_{i}' for i in range(len(importance))],
            'Importance': importance
        })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/feature_importance.png')
    plt.close()
    
    return importance_df

def analyze_with_shap(model, X_test, feature_names=None):
    """
    Analyze model predictions using SHAP values
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    X_test : array-like
        Test features
    feature_names : list
        Names of features
    """
    if not SHAP_AVAILABLE:
        print("SHAP analysis skipped - library not available")
        return
        
    # Create explainer
    explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X_test)
    
    # Plot summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/shap_summary.png')
    plt.close()
    
    # Plot detailed SHAP values for top features
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/shap_bar.png')
    plt.close()

def analyze_with_lime(model, X_train, X_test, y_test, feature_names=None):
    """
    Analyze model predictions using LIME
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    X_train : array-like
        Training features (needed for LIME explainer)
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    feature_names : list
        Names of features
    """
    if not LIME_AVAILABLE:
        print("LIME analysis skipped - library not available")
        return
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train, 
        feature_names=feature_names if feature_names is not None else [f'Feature_{i}' for i in range(X_train.shape[1])],
        class_names=['Not Hired', 'Hired'],
        mode='classification'
    )
    
    # Select a sample to explain
    # Find one positive and one negative example
    pos_indices = np.where(y_test == 1)[0]
    neg_indices = np.where(y_test == 0)[0]
    
    indices_to_explain = []
    if len(pos_indices) > 0:
        indices_to_explain.append(pos_indices[0])
    if len(neg_indices) > 0:
        indices_to_explain.append(neg_indices[0])
    
    # Generate explanations
    for i, idx in enumerate(indices_to_explain):
        # Get explanation
        exp = explainer.explain_instance(
            X_test[idx], 
            model.predict_proba,
            num_features=10
        )
        
        # Plot explanation
        fig = plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure(label=1)  # 1 is the 'Hired' class
        plt.title(f'LIME Explanation for {"Hired" if y_test[idx] == 1 else "Not Hired"} Candidate')
        plt.tight_layout()
        plt.savefig(f'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/lime_explanation_{i+1}.png')
        plt.close()
        
    # Generate a summary of feature importance across multiple samples
    plt.figure(figsize=(12, 8))
    
    # Sample a subset of test instances for the summary
    n_samples = min(50, len(X_test))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Track feature importance
    feature_importance = {}
    
    for idx in sample_indices:
        exp = explainer.explain_instance(
            X_test[idx], 
            model.predict_proba,
            num_features=10
        )
        
        # Extract feature importance
        for feature, importance in exp.as_list(label=1):  # 1 is the 'Hired' class
            if feature in feature_importance:
                feature_importance[feature].append(abs(importance))
            else:
                feature_importance[feature] = [abs(importance)]
    
    # Calculate average importance
    avg_importance = {feature: np.mean(values) for feature, values in feature_importance.items()}
    
    # Sort features by importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top 15 features
    top_features = sorted_features[:15]
    plt.barh([f[0] for f in top_features], [f[1] for f in top_features])
    plt.title('Average Feature Importance (LIME)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/lime_summary.png')
    plt.close()

def save_model(model, model_path, preprocessor=None, preprocessor_path=None):
    """
    Save the trained model and preprocessor
    
    Parameters:
    -----------
    model : object
        Trained model
    model_path : str
        Path to save the model
    preprocessor : object
        Preprocessor
    preprocessor_path : str
        Path to save the preprocessor
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save preprocessor if provided
    if preprocessor is not None and preprocessor_path is not None:
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)

def compare_models(metrics_basic, metrics_text):
    """
    Compare metrics between basic and text-enhanced models
    
    Parameters:
    -----------
    metrics_basic : dict
        Metrics for basic model
    metrics_text : dict
        Metrics for text-enhanced model
    """
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Metric': list(metrics_basic.keys()),
        'Basic Model': list(metrics_basic.values()),
        'Text-Enhanced Model': list(metrics_text.values())
    })
    
    # Calculate improvement
    comparison['Improvement'] = comparison['Text-Enhanced Model'] - comparison['Basic Model']
    comparison['Improvement %'] = (comparison['Improvement'] / comparison['Basic Model'] * 100).round(2)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Create bar positions
    bar_width = 0.35
    r1 = np.arange(len(comparison))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    plt.bar(r1, comparison['Basic Model'], width=bar_width, label='Basic Model')
    plt.bar(r2, comparison['Text-Enhanced Model'], width=bar_width, label='Text-Enhanced Model')
    
    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison: Basic vs Text-Enhanced')
    plt.xticks([r + bar_width/2 for r in range(len(comparison))], comparison['Metric'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/model_comparison.png')
    plt.close()
    
    return comparison

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Preprocess data - basic features
    print("\nPreprocessing data with basic features...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data, text_features=False)
    
    # Feature selection for basic features
    print("\nSelecting important features from basic features...")
    X_train_selected, X_test_selected, selector_basic = select_important_features(X_train, y_train, X_test)
    
    # Save the selector for later use
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/selector_basic.pkl', 'wb') as f:
        pickle.dump(selector_basic, f)
    
    # Save the preprocessed data for later use
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    
    # Train XGBoost model with selected basic features
    print("\nTraining XGBoost model with selected basic features...")
    model_basic, metrics_basic = train_xgboost_model(X_train_selected, y_train, X_test_selected, y_test, use_grid_search=True)
    
    # Print metrics for basic model
    print("\nBasic Model Metrics (with feature selection and tuning):")
    for metric, value in metrics_basic.items():
        print(f"{metric}: {value:.4f}")
    
    # Save basic model
    save_model(
        model_basic, 
        'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/xgboost_basic_improved.pkl',
        preprocessor,
        'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/preprocessor_basic.pkl'
    )
    
    # Preprocess data - with text features
    print("\nPreprocessing data with text features...")
    X_train_text, X_test_text, y_train_text, y_test_text, preprocessor_text = preprocess_data(data, text_features=True)
    
    # Feature selection for text features
    print("\nSelecting important features from text-enhanced features...")
    X_train_text_selected, X_test_text_selected, selector_text = select_important_features(X_train_text, y_train_text, X_test_text)
    
    # Train XGBoost model with selected text features
    print("\nTraining XGBoost model with selected text features...")
    model_text, metrics_text = train_xgboost_model(X_train_text_selected, y_train_text, X_test_text_selected, y_test_text, use_grid_search=True)
    
    # Print metrics for text-enhanced model
    print("\nText-Enhanced Model Metrics (with feature selection and tuning):")
    for metric, value in metrics_text.items():
        print(f"{metric}: {value:.4f}")
    
    # Save text-enhanced model
    save_model(
        model_text, 
        'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/xgboost_text_improved.pkl',
        preprocessor_text,
        'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/preprocessor_text.pkl'
    )
    
    # Compare models
    print("\nComparing improved models...")
    comparison = compare_models(metrics_basic, metrics_text)
    print("\nModel Comparison:")
    print(comparison)
    
    # Analyze feature importance for basic model
    print("\nAnalyzing feature importance for basic model...")
    importance_basic = analyze_feature_importance(model_basic)
    print("\nTop 10 features (basic model):")
    print(importance_basic.head(10))
    
    # Analyze feature importance for text-enhanced model
    print("\nAnalyzing feature importance for text-enhanced model...")
    importance_text = analyze_feature_importance(model_text)
    print("\nTop 10 features (text-enhanced model):")
    print(importance_text.head(10))
    
    # LIME analysis for basic model
    print("\nPerforming feature importance analysis with LIME...")
    if LIME_AVAILABLE:
        try:
            # Use the selected features for LIME analysis instead of original features
            analyze_with_lime(model_basic, X_train_selected, X_test_selected, y_test)
        except Exception as e:
            print(f"LIME analysis failed: {str(e)}")
            print("Falling back to standard feature importance visualization...")
            # Plot confusion matrix as an alternative visualization
            y_pred = model_basic.predict(X_test_selected)  # Use selected features here too
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/confusion_matrix.png')
            plt.close()
    else:
        print("LIME analysis skipped - library not available")
        # Plot confusion matrix as an alternative visualization
        y_pred = model_basic.predict(X_test_selected)  # Use selected features here too
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/confusion_matrix.png')
        plt.close()
    
    print("\nDone! Models saved to 'models' directory.")