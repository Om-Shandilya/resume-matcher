import pandas as pd
from pathlib import Path
from src.feature_engg.tfidf_vectorizing_data import load_tfidf_vectorizer, load_tfidf_matrix
from src.feature_engg.bert_embedding_data import load_bert_model, load_faiss_index
from src.processing.text_cleaning import clean_text, clean_text_for_bert
from src.matching.matching_engine import compute_similarity_matrix, top_n_tfidf_matches, top_n_bert_matches

# Defining paths for data files
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
def load_job_titles(job_csv_path: str):
    df = pd.read_csv(job_csv_path)
    if "title" not in df.columns:
        raise ValueError("Job CSV must contain a 'title' column.")
    return df

def run_tfidf_pipeline(raw_resume: str,
                       local_vectorizer_path=None,
                       local_matrix_path=None,
                       repo_id="Om-Shandilya/resume-matcher-tfidf",
                       vectorizer_filename="applicant/job_vectorizer.pkl",
                       matrix_filename="applicant/job_matrix.npz",
                       top_k=None,
                       debug=False):
    """Return top-N matches using TF-IDF pipeline.
    
    Args:
        raw_resume (str): Raw text of the resume.
        local_vectorizer_path (str, optional): Local path to TF-IDF vectorizer.
        local_matrix_path (str, optional): Local path to TF-IDF matrix.
        repo_id (str): Hugging Face repo ID for vectorizer/matrix.
        vectorizer_filename (str): Filename of the vectorizer in the repo.
        matrix_filename (str): Filename of the matrix in the repo.
        top_k (int, optional): Number of top matches to return. If None, return all.
        debug (bool, optional): Print raw similarity scores for both and cleaned resume.
    
    Returns:
        List[Tuple[str, float]]: List of (job_title, score) for top_k matches.
    """
    cleaned_resume = clean_text(raw_resume)

    vectorizer = load_tfidf_vectorizer(local_vectorizer_path, repo_id, vectorizer_filename)
    job_matrix = load_tfidf_matrix(local_matrix_path, repo_id, matrix_filename)

    resume_vector = vectorizer.transform([cleaned_resume])
    sim_matrix = compute_similarity_matrix(resume_vector, job_matrix)

    job_df = load_job_titles(PROJECT_ROOT / "data/app_data/tfidf_job_titles.csv")
    total_jobs = len(job_df['title'].unique())

    message = ""
    if top_k is None:
        final_top_k = total_jobs
        message = f"✅ Showing all {total_jobs} job matches, ranked by relevance."
    elif top_k > total_jobs:
        final_top_k = total_jobs
        message = f"ℹ️ You requested {top_k} matches, but only {total_jobs} are available. Showing all {total_jobs} matches."
    else:
        final_top_k = top_k
        message = f"✅ Showing the top {final_top_k} job matches."

    matches = top_n_tfidf_matches(sim_matrix, top_n=final_top_k, job_df=job_df)
    
    if debug:
        print("\n================ DEBUG MODE ================")
        print("\n📄--- [DEBUG - TFIDF] Cleaned Resume Preview:\n", cleaned_resume[:1000], "---")
        print(f"\n--- [DEBUG - TFIDF] Raw Similarity Scores (top {final_top_k}) ---")
        for job_idx, score in matches[0]:
            print(f"[{job_idx}] {job_df.iloc[job_idx]['title']} → {score:0.6f}")
        print("==============================================")

    return [(job_df.iloc[j]['title'], score) for j, score in matches[0]],message


def run_bert_pipeline(raw_resume: str,
                      local_bert_path=None,
                      local_index_path=None,
                      repo_id="Om-Shandilya/resume-matcher-bert",
                      index_filename="applicant/jobs.faiss",
                      top_k=None,
                      debug=False):
    """Return top-N matches using BERT + FAISS pipeline.
    
    Args:
        raw_resume (str): Raw text of the resume.
        local_bert_path (str, optional): Local path to BERT model. 
        local_index_path (str, optional): Local path to FAISS index.
        repo_id (str): Hugging Face repo ID for model/index.
        index_filename (str): Filename of the FAISS index in the repo.
        top_k (int, optional): Number of top matches to return. If None, return all.
        debug (bool, optional): Print raw similarity scores for both and cleaned resume.
    
    Returns:
        List[Tuple[str, float]]: List of (job_title, score) for top_k matches.
    """
    model = load_bert_model(local_bert_path=local_bert_path, repo_id=repo_id)
    job_index = load_faiss_index(local_index_path, repo_id, index_filename)

    cleaned_resume = clean_text_for_bert(raw_resume)
    resume_embedding = model.encode([cleaned_resume], normalize_embeddings=True)

    D, I = job_index.search(resume_embedding, job_index.ntotal)
    job_df = load_job_titles(PROJECT_ROOT / "data/app_data/bert_job_titles.csv")
    total_jobs = len(job_df['title'].unique())

    message = ""
    if top_k is None:
        final_top_k = total_jobs
        message = f"✅ Showing all {total_jobs} job matches, ranked by relevance."
    elif top_k > total_jobs:
        final_top_k = total_jobs
        message = f"ℹ️ You requested {top_k} matches, but only {total_jobs} are available. Showing all {total_jobs} matches."
    else:
        final_top_k = top_k
        message = f"✅ Showing the top {final_top_k} job matches."

    matches = top_n_bert_matches(I, D, job_df, top_n=final_top_k)

    if debug:
        print("\n================ DEBUG MODE ================")
        print(f"\n--- [DEBUG - BERT/FAISS] Raw Similarity Scores (top {final_top_k}) ---")
        for idx, score in matches:
            print(f"[{idx}] {job_df.iloc[idx]['title']} → {score:0.6f}")
        print("==============================================")

    return [(job_df.iloc[i]['title'], score) for i, score in matches], message
