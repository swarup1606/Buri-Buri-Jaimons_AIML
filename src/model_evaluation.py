import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import xgboost as xgb

from data_preparation import load_data, preprocess_data

def evaluate_with_cross_validation(X, y, model, cv=5):
    """
    Evaluate model using cross-validation
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    model : estimator
        Model to evaluate
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Cross-validation results
    """
    # Define metrics to evaluate
    metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Perform cross-validation for each metric
    cv_results = {}
    for metric_name, scoring in metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    return cv_results

def plot_learning_curve(X, y, model, cv=5):
    """
    Plot learning curve to evaluate model performance with varying training set sizes
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    model : estimator
        Model to evaluate
    cv : int
        Number of cross-validation folds
    """
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc'
    )
    
    # Calculate mean and std for training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('ROC AUC Score')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/learning_curve.png')
    plt.close()

def plot_roc_curve(X, y, model):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    model : estimator
        Trained model
    """
    # Predict probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/roc_curve.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, X_test, y, y_test, preprocessor = preprocess_data(data, text_features=False)
    
    # Create model with best parameters from previous tuning
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=3,
        gamma=0,
        scale_pos_weight=1,
        objective='binary:logistic',
        random_state=42
    )
    
    # Evaluate with cross-validation
    print("\nEvaluating model with cross-validation...")
    cv_results = evaluate_with_cross_validation(X, y, model)
    
    # Print cross-validation results
    print("\nCross-Validation Results:")
    for metric, result in cv_results.items():
        print(f"{metric}: {result['mean']:.4f} ± {result['std']:.4f}")
    
    # Plot learning curve
    print("\nPlotting learning curve...")
    plot_learning_curve(X, y, model)
    
    # Train model on full training data
    print("\nTraining model on full training data...")
    model.fit(X, y)
    
    # Plot ROC curve
    print("\nPlotting ROC curve...")
    plot_roc_curve(X_test, y_test, model)
    
    print("\nDone! Evaluation results saved to project directory.")

if __name__ == "__main__":
    main()