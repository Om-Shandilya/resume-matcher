import pandas as pd

def load_resume_data(path="data/raw/resumes/Resume.csv"):
    df = pd.read_csv(path, encoding='utf-8')                     # 1. Load the CSV file
    df = df[['Category', 'Resume_str']].dropna()                 # 2. Keep only two columns, drop rows with missing values
    df.columns = ['role', 'text']                                # 3. Rename columns to 'role' and 'text'
    return df                                                    # 4. Return the cleaned DataFrame

def load_job_data(path="data/raw/job_descriptions/Job_Descriptions.csv", sample_size=1000):
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)     # 1. Load large CSV with all job descriptions
    df = df[['Job Title', 'Job Description']].dropna()             # 2. Keep only two columns, drop rows with missing values
    df.columns = ['title', 'text']                                 # 3. Rename columns to standard names
    df = df.sample(n=sample_size, random_state=42)                 # 4. Randomly sample 1000 job descriptions
    return df                                                      # 5. Return the cleaned and sampled DataFrame