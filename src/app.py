from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np  # Add NumPy import
import os
import sys
from salary_prediction import predict_salary

# Create a Flask app with the correct template folder path
template_dir = os.path.abspath('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/templates')
app = Flask(__name__, template_folder=template_dir)

# Load model, preprocessor, and selector
def load_model():
    model_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/xgboost_basic_improved.pkl'
    preprocessor_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/preprocessor_basic.pkl'
    selector_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/models/selector_basic.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Try to load selector, but handle the case when it doesn't exist
    selector = None
    if os.path.exists(selector_path):
        with open(selector_path, 'rb') as f:
            selector = pickle.load(f)
    else:
        print("Warning: Selector file not found. Proceeding without feature selection.")
    
    return model, preprocessor, selector

model, preprocessor, selector = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    candidate_data = {
        'years_experience': float(request.form.get('years_experience')),
        'skill_match_score': float(request.form.get('skill_match_score')),
        'education': request.form.get('education'),
        'job_level': request.form.get('job_level'),
        'industry': request.form.get('industry')
    }
    
    # Convert to DataFrame
    candidate_df = pd.DataFrame([candidate_data])
    
    # Preprocess
    X = preprocessor.transform(candidate_df)
    
    try:
        # Apply feature selection if selector is available
        if selector is not None:
            try:
                X_selected = selector.transform(X)
            except IndexError as e:
                print(f"Feature selection error: {str(e)}")
                print(f"Selector expects shape: {len(selector.get_support())}, but got: {X.shape[1]}")
                # Fall back to using the model directly with the preprocessed data
                X_selected = X
        else:
            # If no selector, use the preprocessed data directly
            X_selected = X
        
        # Make prediction
        try:
            prediction = bool(model.predict(X_selected)[0])
            probability = float(model.predict_proba(X_selected)[0, 1])
            
            result = {
                'prediction': prediction,
                'probability': probability,
                'recommendation': 'Recommended' if prediction else 'Not Recommended'
            }
            
            return render_template('result.html', result=result)
        except ValueError as e:
            # If direct prediction fails, try to adapt the features
            if "Feature shape mismatch" in str(e):
                error_message = f"Error: {str(e)}. The model expects a different number of features than provided."
                
                # Try to extract expected feature count from error message
                import re
                match = re.search(r'expected: (\d+), got (\d+)', str(e))
                if match:
                    expected_features = int(match.group(1))
                    got_features = int(match.group(2))
                    
                    # If we have more features than expected, truncate
                    if got_features > expected_features and hasattr(X_selected, 'shape'):
                        print(f"Attempting to truncate features from {got_features} to {expected_features}")
                        if isinstance(X_selected, np.ndarray):
                            X_selected = X_selected[:, :expected_features]
                        else:
                            # For sparse matrices
                            X_selected = X_selected[:, :expected_features]
                        
                        # Try prediction again
                        try:
                            prediction = bool(model.predict(X_selected)[0])
                            probability = float(model.predict_proba(X_selected)[0, 1])
                            
                            result = {
                                'prediction': prediction,
                                'probability': probability,
                                'recommendation': 'Recommended' if prediction else 'Not Recommended'
                            }
                            
                            return render_template('result.html', result=result)
                        except Exception as inner_e:
                            error_message += f"\nAttempted feature adaptation failed: {str(inner_e)}"
                
                return render_template('error.html', error=error_message)
            else:
                raise
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        return render_template('error.html', error=error_message)

@app.route('/salary', methods=['GET'])
def salary_form():
    return render_template('salary_form.html')

@app.route('/predict_salary', methods=['POST'])
def salary_prediction():
    try:
        # Get form data
        candidate_data = {
            'years_experience': float(request.form.get('years_experience')),
            'education': request.form.get('education'),
            'job_level': request.form.get('job_level'),
            'industry': request.form.get('industry'),
            'location': request.form.get('location', 'Unknown')
        }
        
        # Predict salary
        predicted_salary = predict_salary(candidate_data)
        
        if predicted_salary is None:
            return render_template('error.html', error="Failed to predict salary. Please try again.")
        
        # Format salary for display
        formatted_salary = "${:,.2f}".format(predicted_salary)
        
        # Get chart data
        chart_data = get_salary_chart_data()
        
        result = {
            'predicted_salary': formatted_salary,
            'candidate_data': candidate_data,
            'chart_data': chart_data
        }
        
        return render_template('salary_result.html', result=result)
    
    except Exception as e:
        error_message = f"Salary prediction error: {str(e)}"
        print(f"Error in salary prediction: {error_message}")  # Add this for debugging
        return render_template('error.html', error=error_message)

def get_salary_chart_data():
    """
    Generate data for salary charts based on the trained model
    """
    try:
        # Load the salary data
        data_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/data/Salary_data.csv'
        df = pd.read_csv(data_path)
        
        # Experience vs Salary data
        exp_ranges = [0, 5, 10, 15, 20, 25, 30]
        exp_salaries = []
        
        for i in range(len(exp_ranges)-1):
            lower = exp_ranges[i]
            upper = exp_ranges[i+1]
            avg_salary = df[(df['years_experience'] >= lower) & 
                           (df['years_experience'] < upper)]['salary'].mean()
            exp_salaries.append(int(avg_salary))
        
        # Add the last range
        avg_salary = df[df['years_experience'] >= exp_ranges[-1]]['salary'].mean()
        exp_salaries.append(int(avg_salary))
        
        # Education data
        edu_salaries = []
        education_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
        
        for edu in education_levels:
            avg_salary = df[df['education'] == edu]['salary'].mean()
            edu_salaries.append(int(avg_salary))
        
        # Industry data
        industry_salaries = []
        industries = ['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'Retail']
        
        for industry in industries:
            avg_salary = df[df['industry'] == industry]['salary'].mean()
            industry_salaries.append(int(avg_salary))
        
        # Job level data
        level_salaries = []
        job_levels = ['Entry-level', 'Mid-level', 'Senior', 'Executive']
        
        for level in job_levels:
            avg_salary = df[df['job_level'] == level]['salary'].mean()
            level_salaries.append(int(avg_salary))
        
        return {
            'experience': {
                'labels': exp_ranges,
                'data': exp_salaries
            },
            'education': {
                'labels': education_levels,
                'data': edu_salaries
            },
            'industry': {
                'labels': industries,
                'data': industry_salaries
            },
            'job_level': {
                'labels': job_levels,
                'data': level_salaries
            }
        }
    except Exception as e:
        print(f"Error generating chart data: {str(e)}")
        # Return default data if there's an error
        return {
            'experience': {
                'labels': [0, 5, 10, 15, 20, 25, 30],
                'data': [40000, 60000, 80000, 100000, 120000, 135000, 150000]
            },
            'education': {
                'labels': ['High School', 'Associate', 'Bachelor', 'Master', 'PhD'],
                'data': [45000, 55000, 75000, 95000, 115000]
            },
            'industry': {
                'labels': ['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'Retail'],
                'data': [95000, 90000, 85000, 65000, 75000, 55000]
            },
            'job_level': {
                'labels': ['Entry-level', 'Mid-level', 'Senior', 'Executive'],
                'data': [50000, 75000, 110000, 160000]
            }
        }

if __name__ == '__main__':
    # Create templates directory in the correct location for Flask
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(os.path.dirname(current_dir), 'src', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create HTML templates in the Flask-expected location
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Job Candidate Predictor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; }
                input, select { width: 100%; padding: 8px; box-sizing: border-box; }
                button { background: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Job Candidate Predictor</h1>
                <form action="/predict" method="post">
                    <div class="form-group">
                        <label for="years_experience">Years of Experience</label>
                        <input type="number" name="years_experience" id="years_experience" step="0.5" min="0" required>
                    </div>
                    <div class="form-group">
                        <label for="skill_match_score">Skill Match Score (0-1)</label>
                        <input type="number" name="skill_match_score" id="skill_match_score" step="0.01" min="0" max="1" required>
                    </div>
                    <div class="form-group">
                        <label for="education">Education</label>
                        <select name="education" id="education">
                            <option value="High School">High School</option>
                            <option value="Associate">Associate</option>
                            <option value="Bachelor">Bachelor</option>
                            <option value="Master">Master</option>
                            <option value="PhD">PhD</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="job_level">Job Level</label>
                        <select name="job_level" id="job_level">
                            <option value="Entry-level">Entry-level</option>
                            <option value="Mid-level">Mid-level</option>
                            <option value="Senior">Senior</option>
                            <option value="Executive">Executive</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="industry">Industry</label>
                        <select name="industry" id="industry">
                            <option value="Technology">Technology</option>
                            <option value="Finance">Finance</option>
                            <option value="Healthcare">Healthcare</option>
                            <option value="Education">Education</option>
                            <option value="Manufacturing">Manufacturing</option>
                            <option value="Retail">Retail</option>
                        </select>
                    </div>
                    <button type="submit">Predict</button>
                </form>
            </div>
        </body>
        </html>
        ''')
    
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/templates/result.html', 'w', encoding='utf-8') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .result { padding: 20px; border-radius: 5px; margin-top: 20px; }
                .recommended { background-color: #dff0d8; border: 1px solid #d6e9c6; }
                .not-recommended { background-color: #f2dede; border: 1px solid #ebccd1; }
                a { display: inline-block; margin-top: 20px; color: #337ab7; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result</h1>
                <div class="result {% if result.prediction %}recommended{% else %}not-recommended{% endif %}">
                    <h2>{{ result.recommendation }}</h2>
                    <p>Confidence: {{ "%.2f"|format(result.probability * 100) }}%</p>
                </div>
                <a href="/">Back to Prediction Form</a>
            </div>
        </body>
        </html>
        ''')
    
    # Create error template
    with open('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/templates/error.html', 'w', encoding='utf-8') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .error { padding: 20px; border-radius: 5px; margin-top: 20px; background-color: #f2dede; border: 1px solid #ebccd1; }
                a { display: inline-block; margin-top: 20px; color: #337ab7; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Error</h1>
                <div class="error">
                    <h2>Something went wrong</h2>
                    <p>{{ error }}</p>
                </div>
                <a href="/">Back to Prediction Form</a>
            </div>
        </body>
        </html>
        ''')
    
    # Create salary form template
    with open(os.path.join(templates_dir, 'salary_form.html'), 'w', encoding='utf-8') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Salary Predictor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; }
                input, select { width: 100%; padding: 8px; box-sizing: border-box; }
                button { background: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
                .nav { margin-bottom: 20px; }
                .nav a { margin-right: 15px; color: #337ab7; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="nav">
                    <a href="/">Candidate Predictor</a>
                    <a href="/salary">Salary Predictor</a>
                </div>
                <h1>Salary Predictor</h1>
                <form action="/predict_salary" method="post">
                    <div class="form-group">
                        <label for="years_experience">Years of Experience</label>
                        <input type="number" name="years_experience" id="years_experience" step="0.5" min="0" required>
                    </div>
                    <div class="form-group">
                        <label for="education">Education</label>
                        <select name="education" id="education">
                            <option value="High School">High School</option>
                            <option value="Associate">Associate</option>
                            <option value="Bachelor">Bachelor</option>
                            <option value="Master">Master</option>
                            <option value="PhD">PhD</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="job_level">Job Level</label>
                        <select name="job_level" id="job_level">
                            <option value="Entry-level">Entry-level</option>
                            <option value="Mid-level">Mid-level</option>
                            <option value="Senior">Senior</option>
                            <option value="Executive">Executive</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="industry">Industry</label>
                        <select name="industry" id="industry">
                            <option value="Technology">Technology</option>
                            <option value="Finance">Finance</option>
                            <option value="Healthcare">Healthcare</option>
                            <option value="Education">Education</option>
                            <option value="Manufacturing">Manufacturing</option>
                            <option value="Retail">Retail</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="location">Location</label>
                        <select name="location" id="location">
                            <option value="Urban">Urban</option>
                            <option value="Suburban">Suburban</option>
                            <option value="Rural">Rural</option>
                        </select>
                    </div>
                    <button type="submit">Predict Salary</button>
                </form>
            </div>
        </body>
        </html>
        ''')
    
    # Create salary result template
    with open(os.path.join(templates_dir, 'salary_result.html'), 'w', encoding='utf-8') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Salary Prediction Result</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .result { padding: 20px; border-radius: 5px; margin-top: 20px; background-color: #dff0d8; border: 1px solid #d6e9c6; }
                .details { margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
                .details h3 { margin-top: 0; }
                .details p { margin: 5px 0; }
                a { display: inline-block; margin-top: 20px; color: #337ab7; text-decoration: none; }
                .nav { margin-bottom: 20px; }
                .nav a { margin-right: 15px; color: #337ab7; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="nav">
                    <a href="/">Candidate Predictor</a>
                    <a href="/salary">Salary Predictor</a>
                </div>
                <h1>Salary Prediction Result</h1>
                <div class="result">
                    <h2>Estimated Salary: {{ result.predicted_salary }}</h2>
                </div>
                <div class="details">
                    <h3>Candidate Details</h3>
                    <p><strong>Years of Experience:</strong> {{ result.candidate_data.years_experience }}</p>
                    <p><strong>Education:</strong> {{ result.candidate_data.education }}</p>
                    <p><strong>Job Level:</strong> {{ result.candidate_data.job_level }}</p>
                    <p><strong>Industry:</strong> {{ result.candidate_data.industry }}</p>
                    <p><strong>Location:</strong> {{ result.candidate_data.location }}</p>
                </div>
                <a href="/salary">Back to Salary Predictor</a>
            </div>
        </body>
        </html>
        ''')
    
    # Update index.html to include navigation
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Job Candidate Predictor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; }
                input, select { width: 100%; padding: 8px; box-sizing: border-box; }
                button { background: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
                .nav { margin-bottom: 20px; }
                .nav a { margin-right: 15px; color: #337ab7; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="nav">
                    <a href="/">Candidate Predictor</a>
                    <a href="/salary">Salary Predictor</a>
                </div>
                <h1>Job Candidate Predictor</h1>
                <form action="/predict" method="post">
                    <div class="form-group">
                        <label for="years_experience">Years of Experience</label>
                        <input type="number" name="years_experience" id="years_experience" step="0.5" min="0" required>
                    </div>
                    <div class="form-group">
                        <label for="skill_match_score">Skill Match Score (0-1)</label>
                        <input type="number" name="skill_match_score" id="skill_match_score" step="0.01" min="0" max="1" required>
                    </div>
                    <div class="form-group">
                        <label for="education">Education</label>
                        <select name="education" id="education">
                            <option value="High School">High School</option>
                            <option value="Associate">Associate</option>
                            <option value="Bachelor">Bachelor</option>
                            <option value="Master">Master</option>
                            <option value="PhD">PhD</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="job_level">Job Level</label>
                        <select name="job_level" id="job_level">
                            <option value="Entry-level">Entry-level</option>
                            <option value="Mid-level">Mid-level</option>
                            <option value="Senior">Senior</option>
                            <option value="Executive">Executive</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="industry">Industry</label>
                        <select name="industry" id="industry">
                            <option value="Technology">Technology</option>
                            <option value="Finance">Finance</option>
                            <option value="Healthcare">Healthcare</option>
                            <option value="Education">Education</option>
                            <option value="Manufacturing">Manufacturing</option>
                            <option value="Retail">Retail</option>
                        </select>
                    </div>
                    <button type="submit">Predict</button>
                </form>
            </div>
        </body>
        </html>
        ''')
    
    app.run(debug=True)