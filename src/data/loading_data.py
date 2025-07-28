import pandas as pd

def load_resume_data(path="../../data/raw/resumes/Resume.csv"):
    """
    Loads desired resume data from a CSV file.
    
    Args:
        path (str): Path to the CSV file containing resume data.
        
    Returns:
        pd.DataFrame: A DataFrame with meaningful resume data.
    """
    # 1. Load the CSV file
    df = pd.read_csv(path, encoding='utf-8')

    # 2. Keep only two columns, drop rows with missing values
    df = df[['Category', 'Resume_str']].dropna()

    # 3. Rename columns to 'role' and 'text'          
    df.columns = ['role', 'text']  

    return df                                                   

def load_job_data(path="../../data/raw/job_descriptions/Job_Descriptions.csv", 
                  sample_size=None,
                  resume_count=None):
    """
    Loads desired data and samples it for job description data from a CSV file.
    
    Args:
        path (str): Path to the CSV file containing job description data.
        sample_size (int): Number of job descriptions to sample.
        resume_count (int): If specified, sample this many job descriptions.
    
    Returns:
        pd.DataFrame: A DataFrame with meaningful and sampled job description data.
    """
    # 1. Load large CSV with all job descriptions
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)

    # 2. Keep only two columns, drop rows with missing values 
    df = df[['Job Title', 'Job Description']].dropna()

    # 3. Rename columns to standard names
    df.columns = ['title', 'text']

    # 4. If sample_size is None, use resume_count is not None set it as sample_size
    if sample_size is None and resume_count is not None:
        sample_size = resume_count

    # 5. Randomly sample job descriptions if sample_size is specified
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)

    return df