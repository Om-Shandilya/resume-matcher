import pandas as pd

def make_combined_corpus(resume_path, job_path, total_samples=100_000, random_state=42):
    """
    Makes a combined corpus of resume and job descriptions with stratified sampling based on job titles.

    Args:
        resume_path (str): Path to the resume CSV file.
        job_path (str): Path to the job CSV file.
        total_samples (int): Total number of samples to make. (default=100_000)
        random_state (int): Random seed for reproducibility. (default=42)
    
    Returns:
        pd.DataFrame: Combined corpus and sampled job descriptions.
    """
    # Load data
    df_resumes = pd.read_csv(resume_path)
    df_jobs = pd.read_csv(job_path)

    # Stratified sampling
    n_titles = df_jobs['title'].nunique()
    samples_per_title = max(1, total_samples // n_titles)

    df_jobs_sample = (
        df_jobs.groupby("title", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), samples_per_title), random_state=random_state))
        .reset_index(drop=True)
    )

    combined_corpus = pd.concat([
        df_resumes['text_cleaned'],
        df_jobs_sample['text_cleaned']
    ], ignore_index=True)

    return combined_corpus, df_jobs_sample

if __name__ == "__main__":
    combined_corpus, df_jobs_sample = make_combined_corpus(
        "data/processed/resumes_cleaned.csv",
        "data/processed/all_jds_cleaned.csv",
        total_samples=100_000
    )

    combined_corpus.to_csv("data/processed/combined_corpus.csv", index=False)
    df_jobs_sample.to_csv("data/processed/jobs_sampled.csv", index=False)
