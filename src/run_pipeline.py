import pandas as pd
from datetime import datetime
import os
from src.data.loading_data import load_job_data, load_resume_data
from src.processing.text_cleaning import clean_column
from src.data.saving_data import save_dataset

# Step 1: Load data
df_resumes = load_resume_data()
df_jobs = load_job_data(resume_count = len(df_resumes))

# Step 2: Clean resume texts
df_resumes = clean_column(
    df=df_resumes,
    column_name='text',
    new_column_name='text_cleaned',
    remove_numbers = True,
    remove_stopwords = True,
    apply_lemmatization = True)
    

# Step 3: Clean job descriptions
df_jobs = clean_column(
    df=df_jobs,
    column_name='text',
    new_column_name='text_cleaned',
    remove_numbers = True,
    remove_stopwords = True,
    apply_lemmatization = True
)

# Step 4: Save 5 latest versions of cleaned data with timestamp

save_dataset(df_resumes, base_filename="resumes_cleaned", max_versions=5)
save_dataset(df_jobs, base_filename="job_descriptions_cleaned", max_versions=5)