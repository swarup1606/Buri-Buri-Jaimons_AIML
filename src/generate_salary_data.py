import pandas as pd
import numpy as np
import os

def generate_salary_data(n_samples=1000):
    """
    Generate synthetic salary data for testing
    """
    # Create data directory if it doesn't exist
    os.makedirs('c:/Users/Vaibhav/OneDrive/Desktop/predict_job/data', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define possible values for categorical features
    education_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
    job_levels = ['Entry-level', 'Mid-level', 'Senior', 'Executive']
    industries = ['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'Retail']
    locations = ['Urban', 'Suburban', 'Rural']
    
    # Generate random data
    data = {
        'years_experience': np.random.uniform(0, 30, n_samples),
        'education': np.random.choice(education_levels, n_samples),
        'job_level': np.random.choice(job_levels, n_samples),
        'industry': np.random.choice(industries, n_samples),
        'location': np.random.choice(locations, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate salary based on features (with some randomness)
    # Base salary
    base_salary = 30000
    
    # Experience factor (more experience = higher salary)
    exp_factor = df['years_experience'] * 2000
    
    # Education factor
    edu_mapping = {
        'High School': 0,
        'Associate': 5000,
        'Bachelor': 15000,
        'Master': 25000,
        'PhD': 35000
    }
    edu_factor = df['education'].map(edu_mapping)
    
    # Job level factor
    level_mapping = {
        'Entry-level': 0,
        'Mid-level': 20000,
        'Senior': 40000,
        'Executive': 80000
    }
    level_factor = df['job_level'].map(level_mapping)
    
    # Industry factor
    industry_mapping = {
        'Technology': 15000,
        'Finance': 12000,
        'Healthcare': 10000,
        'Education': 5000,
        'Manufacturing': 8000,
        'Retail': 3000
    }
    industry_factor = df['industry'].map(industry_mapping)
    
    # Location factor
    location_mapping = {
        'Urban': 10000,
        'Suburban': 5000,
        'Rural': 0
    }
    location_factor = df['location'].map(location_mapping)
    
    # Calculate salary with some random noise
    df['salary'] = (base_salary + exp_factor + edu_factor + level_factor + 
                   industry_factor + location_factor + 
                   np.random.normal(0, 10000, n_samples))
    
    # Ensure salary is positive
    df['salary'] = np.maximum(df['salary'], 20000)
    
    # Save to CSV
    output_path = 'c:/Users/Vaibhav/OneDrive/Desktop/predict_job/data/Salary_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated {n_samples} salary data samples and saved to {output_path}")
    return df

if __name__ == "__main__":
    # Generate 1000 samples by default
    df = generate_salary_data(1000)
    
    # Display some statistics about the generated data
    print("\nData Statistics:")
    print(f"Average Salary: ${df['salary'].mean():.2f}")
    print(f"Minimum Salary: ${df['salary'].min():.2f}")
    print(f"Maximum Salary: ${df['salary'].max():.2f}")
    
    # Display salary distribution by education level
    print("\nAverage Salary by Education Level:")
    for edu in sorted(df['education'].unique()):
        avg_salary = df[df['education'] == edu]['salary'].mean()
        print(f"{edu}: ${avg_salary:.2f}")
    
    # Display salary distribution by job level
    print("\nAverage Salary by Job Level:")
    for level in sorted(df['job_level'].unique()):
        avg_salary = df[df['job_level'] == level]['salary'].mean()
        print(f"{level}: ${avg_salary:.2f}")
    
    # Display salary distribution by industry
    print("\nAverage Salary by Industry:")
    for industry in sorted(df['industry'].unique()):
        avg_salary = df[df['industry'] == industry]['salary'].mean()
        print(f"{industry}: ${avg_salary:.2f}")
    
    print("\nSalary data generation complete!")