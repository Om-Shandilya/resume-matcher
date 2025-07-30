import sys
from data.loading_data import load_or_clean_resume_data, load_or_clean_job_data
from data.saving_data import save_dataset



# Step 1: Load and Clean resume texts
df_resumes = load_or_clean_resume_data()
    

# Step 2: Load and Clean job description texts
df_jobs = load_or_clean_job_data(sample_size=len(df_resumes))

# Step 3: Save 5 latest versions of cleaned data with timestamp

save_dataset(df_resumes, base_filename="resumes_cleaned", max_versions=5)
save_dataset(df_jobs, base_filename="job_descriptions_cleaned", max_versions=5)

# Step 4: Visualize cleaned and processed data

# In order to visualize the data, we can run the EDA script as "python pipeline.py --eda" in bash
if __name__ == "__main__":
    # Set to True if you want EDA plots
    RUN_EDA = "--eda" in sys.argv

    if RUN_EDA:
        from visualization.run_eda import visualize_cleaned_data
        visualize_cleaned_data(df_resumes, df_jobs, save_plots=False)

