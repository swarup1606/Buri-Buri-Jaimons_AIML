import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# Load model, preprocessor, and data
def load_data():
    model_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/xgboost_basic_improved.pkl'
    preprocessor_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/preprocessor_basic.pkl'
    selector_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/selector_basic.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(selector_path, 'rb') as f:
        selector = pickle.load(f)
    
    # Load test data if available
    try:
        with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        
        with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
    except:
        X_test = None
        y_test = None
    
    return model, preprocessor, selector, X_test, y_test

# Set page title
st.set_page_config(page_title="Job Candidate Prediction Dashboard", layout="wide")

# Title
st.title("Job Candidate Prediction Dashboard")

# Load data
model, preprocessor, selector, X_test, y_test = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Overview", "Feature Importance", "Make Prediction"])

if page == "Model Overview":
    st.header("Model Overview")
    
    # Display model parameters
    st.subheader("Model Parameters")
    params = model.get_params()
    st.json(params)
    
    # Display model performance metrics if test data is available
    if X_test is not None and y_test is not None:
        st.subheader("Model Performance")
        
        # Apply feature selection
        X_test_selected = selector.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_pred_proba)
        }
        
        # Display metrics
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        st.table(metrics_df)
        
        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
        
        # Plot ROC curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')
        st.pyplot(fig)

elif page == "Feature Importance":
    st.header("Feature Importance")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for visualization
    feature_names = [f'Feature_{i}' for i in range(len(importance))]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Display top features
    st.subheader("Top 10 Most Important Features")
    st.table(importance_df.head(10))
    
    # Plot feature importance
    st.subheader("Feature Importance Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), ax=ax)
    ax.set_title('Top 20 Feature Importance')
    st.pyplot(fig)

elif page == "Make Prediction":
    st.header("Make a Prediction")
    
    # Create input form
    st.subheader("Enter Candidate Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        years_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)
        skill_match_score = st.slider("Skill Match Score", min_value=0.0, max_value=1.0, step=0.01)
    
    with col2:
        education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "PhD"])
        job_level = st.selectbox("Job Level", ["Entry-level", "Mid-level", "Senior", "Executive"])
        industry = st.selectbox("Industry", ["Technology", "Finance", "Healthcare", "Education", "Manufacturing", "Retail"])
    
    # Make prediction when button is clicked
    if st.button("Predict"):
        # Create candidate data
        candidate_data = {
            'years_experience': years_experience,
            'skill_match_score': skill_match_score,
            'education': education,
            'job_level': job_level,
            'industry': industry
        }
        
        # Convert to DataFrame
        candidate_df = pd.DataFrame([candidate_data])
        
        # Preprocess
        X = preprocessor.transform(candidate_df)
        
        # Apply feature selection
        X_selected = selector.transform(X)
        
        # Make prediction
        prediction = bool(model.predict(X_selected)[0])
        probability = float(model.predict_proba(X_selected)[0, 1])
        
        # Display result
        st.subheader("Prediction Result")
        
        if prediction:
            st.success(f"Recommended (Confidence: {probability:.2%})")
        else:
            st.error(f"Not Recommended (Confidence: {1-probability:.2%})")
        
        # Display feature contributions
        st.subheader("Feature Contributions")
        
        # Get feature importance for this prediction
        feature_names = [f'Feature_{i}' for i in range(X_selected.shape[1])]
        contributions = model.feature_importances_ * X_selected[0]
        
        # Create DataFrame for visualization
        contributions_df = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': contributions
        })
        
        # Sort by absolute contribution
        contributions_df['Abs_Contribution'] = contributions_df['Contribution'].abs()
        contributions_df = contributions_df.sort_values('Abs_Contribution', ascending=False).drop('Abs_Contribution', axis=1)
        
        # Display top contributions
        st.table(contributions_df.head(5))

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Job Candidate Prediction Dashboard v1.0")