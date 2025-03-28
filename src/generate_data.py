import pandas as pd
import numpy as np
import os

def generate_sample_data(n_samples=1000, output_path='data/hiring_data.csv'):
    """
    Generate sample hiring data based on the project requirements
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    output_path : str
        Path to save the generated data
    """
    np.random.seed(42)
    
    # Define possible values for categorical features
    skills_list = ['Python', 'Java', 'SQL', 'Machine Learning', 'Data Analysis', 
                  'JavaScript', 'C++', 'Cloud Computing', 'DevOps', 'Project Management',
                  'R', 'Tableau', 'Power BI', 'Excel', 'Statistics', 'Deep Learning',
                  'NLP', 'Computer Vision', 'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes']
    
    education_list = ['Bachelor\'s', 'Master\'s', 'PhD', 'High School', 'Associate\'s']
    
    job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer',
                 'Project Manager', 'UX Designer', 'QA Engineer', 'Business Analyst',
                 'Data Engineer', 'Machine Learning Engineer', 'Full Stack Developer',
                 'Frontend Developer', 'Backend Developer', 'Cloud Architect']
    
    certifications = ['AWS Certified', 'PMP', 'Scrum Master', 'CISSP', 'Google Cloud', 
                     'Microsoft Certified', 'Cisco Certified', 'CompTIA', 'None']
    
    job_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director']
    
    industries = ['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Education',
                 'Government', 'Consulting', 'Energy', 'Transportation']
    
    # Generate random data
    data = {
        'skills': [', '.join(np.random.choice(skills_list, size=np.random.randint(1, 6), replace=False)) 
                  for _ in range(n_samples)],
        'education': np.random.choice(education_list, size=n_samples),
        'years_experience': np.random.uniform(0, 15, size=n_samples).round(1),
        'past_job_titles': [', '.join(np.random.choice(job_titles, size=np.random.randint(1, 4), replace=False)) 
                           for _ in range(n_samples)],
        'certifications': np.random.choice(certifications, size=n_samples),
        'required_skills': [', '.join(np.random.choice(skills_list, size=np.random.randint(1, 5), replace=False)) 
                           for _ in range(n_samples)],
        'job_title': np.random.choice(job_titles, size=n_samples),
        'job_level': np.random.choice(job_levels, size=n_samples),
        'industry': np.random.choice(industries, size=n_samples),
    }
    
    # Calculate skill match score (feature engineering)
    data['skill_match_score'] = [calculate_skill_match(data['skills'][i], data['required_skills'][i]) 
                                for i in range(n_samples)]
    
    # Generate target variable based on features
    # Higher skill match, more experience, and higher education level increase chances of being hired
    education_score = pd.Series(data['education']).map({
        'High School': 0.5, 'Associate\'s': 0.6, 'Bachelor\'s': 0.7, 'Master\'s': 0.8, 'PhD': 0.9
    }).values
    
    experience_score = np.minimum(data['years_experience'] / 10, 1.0)
    
    # Calculate probability of being hired
    hire_prob = (0.4 * np.array(data['skill_match_score']) + 
                0.3 * experience_score + 
                0.3 * education_score)
    
    # Add some randomness
    hire_prob = np.clip(hire_prob + np.random.normal(0, 0.1, size=n_samples), 0, 1)
    
    # Generate binary target
    data['hired'] = (hire_prob > 0.5).astype(int)
    
    # Generate performance score for hired candidates
    performance_score = np.zeros(n_samples)
    hired_indices = np.where(data['hired'] == 1)[0]
    performance_score[hired_indices] = np.clip(hire_prob[hired_indices] * 10 + 
                                             np.random.normal(0, 1, size=len(hired_indices)), 1, 10)
    data['performance_score'] = performance_score.round(1)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Generated {n_samples} samples and saved to {output_path}")
    
    return df

def calculate_skill_match(candidate_skills, job_skills):
    """
    Calculate the skill match score between candidate and job
    
    Parameters:
    -----------
    candidate_skills : str
        Comma-separated list of candidate skills
    job_skills : str
        Comma-separated list of required job skills
        
    Returns:
    --------
    float
        Skill match score between 0 and 1
    """
    if not candidate_skills or not job_skills:
        return 0.0
    
    # Convert to sets for easier comparison
    candidate_set = set(s.strip().lower() for s in candidate_skills.split(','))
    job_set = set(s.strip().lower() for s in job_skills.split(','))
    
    # Calculate match score
    matched_skills = candidate_set.intersection(job_set)
    match_score = len(matched_skills) / len(job_set) if job_set else 0
    
    return match_score

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(n_samples=5000, output_path='c:/Users/Vaibhav/OneDrive/Desktop/predict_job/data/hiring_data.csv')
    
    # Print sample statistics
    print("\nData Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Hired candidates: {df['hired'].sum()} ({df['hired'].mean()*100:.1f}%)")
    print(f"Average years of experience: {df['years_experience'].mean():.1f}")
    print(f"Average skill match score: {df['skill_match_score'].mean():.2f}")
    print(f"Average performance score (hired only): {df.loc[df['hired']==1, 'performance_score'].mean():.1f}")
    
    # Print sample of the data
    print("\nSample data:")
    print(df.head())