import sys
import os
import pandas as pd
from src.data.loading_data import load_or_clean_resume_data, load_or_clean_job_data
from src.data.saving_data import save_dataset


# Run this script to execute the entire pipeline: 
# Using "python -m src.run_pipeline.py" in root directory


# Step 1: Load and Clean resume texts
df_resumes = load_or_clean_resume_data(cleaned_path="data/processed/resumes_cleaned.csv", 
                                       raw_path="data/raw/resumes/Resume.csv")
    

# Step 2: Load and Clean job description texts
df_jobs = load_or_clean_job_data(cleaned_path="data/processed/jobs_cleaned.csv", 
                                 raw_path="data/raw/resumes/Resume.csv",
                                 sample_size=len(df_resumes))

# Step 3: Save 2 latest versions of cleaned data with timestamp
save_dataset(df_resumes, base_filename="resumes_cleaned", max_versions=1)
save_dataset(df_jobs, base_filename="jobs_cleaned", max_versions=1)

# Step 4: Visualize cleaned and processed data
# Note: The visualization is optional and can be run separately.
# In order to visualize the data, we can run the EDA script as "python run_pipeline.py --eda" in bash

if __name__ == "__main__":
    # Set to True if you want EDA plots
    RUN_EDA = "--eda" in sys.argv

    if RUN_EDA:
        from visualization.run_eda import visualize_cleaned_data
        visualize_cleaned_data(df_resumes, df_jobs, save_plots=False)


# Step 4: Vectorize using shared TF-IDF vectorizer
from src.feature_engg.vectorizing_data import (
    get_tfidf_vectorizer, vectorize_text, save_vectorizer, save_vector_data
)

print("\n💻 Vectorizing text using shared TF-IDF vectorizer...")

# Combine data for fitting the shared vectorizer
combined_corpus = pd.concat([df_resumes["text_cleaned"], df_jobs["text_cleaned"]])
combined_corpus = combined_corpus.dropna()
shared_vectorizer = get_tfidf_vectorizer()
shared_vectorizer.fit(combined_corpus)

# Create directories if they don't exist
vector_save_dir = "models/dev_tfidf"
os.makedirs(vector_save_dir, exist_ok=True)

# Transform resumes and jobs separately using the same vectorizer
X_resumes, _ = vectorize_text(
    df_resumes, text_column="text_cleaned", label="resumes",
    vectorizer=shared_vectorizer, fit_vectorizer=False,
    save_path=vector_save_dir, save_vectorizer_file=False  # We’ll save manually below
)

X_jobs, _ = vectorize_text(
    df_jobs, text_column="text_cleaned", label="jobs",
    vectorizer=shared_vectorizer, fit_vectorizer=False,
    save_path=vector_save_dir, save_vectorizer_file=False
)

# Save shared vectorizer once
save_vectorizer(shared_vectorizer, os.path.join(vector_save_dir, "shared_tfidf_vectorizer.pkl"))

# Step 5: Compute similarity matrix and top matches
from src.matching.matching_engine import compute_similarity_matrix, top_n_matches

print("\n🔍 Computing similarity matrix...")
similarity_matrix = compute_similarity_matrix(X_resumes, X_jobs)

print("\n🏆 Finding top 5 matches for each resume...")
top_matches = top_n_matches(similarity_matrix, top_n=5)

# Display results for first few resumes
for resume_idx, job_matches in list(top_matches.items())[:5]:
    print(f"\n🔎 Resume {resume_idx + 1} best matches:")
    for job_idx, score in job_matches:
        print(f"   ↳ Job {job_idx + 1} with similarity score: {score}")
