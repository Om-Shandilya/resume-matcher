import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.feature_engg.tfidf_vectorizing_data import load_tfidf_matrix

def compute_similarity_matrix(X_resumes, X_jobs ):
    """
    Compute cosine similarity between resume vectors and job vectors.
    Returns a similarity matrix of shape (num_resumes, num_jobs).
    """
    return cosine_similarity(X_resumes, X_jobs)

def top_n_tfidf_matches(similarity_matrix: np.ndarray, 
                  top_n: int = 5,
                  job_df = None):
    """
    For each resume, get indices of top N matching jobs.

    Args:
        similarity_matrix: np.ndarray -> Cosine similarity matrix of shape (num_resumes, num_jobs)
        top_n: int -> Number of top matches to return for each resume.
    
    Returns:
        Dict[int, List[Tuple[job_idx, score]]] -> top job matches per resume
    """
    results = {}
    for i, row in enumerate(similarity_matrix):
        seen_titles = set()
        ranked = []
        for j in row.argsort()[::-1]:  # Sorted indices in descending order
            title = job_df.iloc[j]['title'] if job_df is not None else j
            if title not in seen_titles:
                ranked.append((j, round(row[j], 4)))
                seen_titles.add(title)
            if len(ranked) == top_n:
                break
        results[i] = ranked
    return results

def top_n_bert_matches(indices, distances, job_df, top_n=5):
    """
    Deduplicate FAISS results by job title and return top-N unique matches.
    Searches across all jobs if provided.

    Args:
        indices (np.ndarray): Indices of nearest neighbors from FAISS (shape: [1, k]).
        distances (np.ndarray): Distances/similarities from FAISS (shape: [1, k]).
        job_df (pd.DataFrame): DataFrame containing job titles.
        top_n (int): Number of unique top matches to return.

    Returns:
        List[Tuple[int, float]]: List of (job_idx, score) for top-N unique titles.
    """
    seen_titles = set()
    ranked = []

    for idx, score in zip(indices[0], distances[0]):
        title = job_df.iloc[idx]['title']
        if title not in seen_titles:
            ranked.append((idx, float(score)))
            seen_titles.add(title)
        if len(ranked) == top_n:
            break

    return ranked


if __name__ == "__main__":
    # Define paths
    resume_vec_path = "models/dev_tfidf/resumes_tfidf_matrix.npz"
    job_vec_path = "models/dev_tfidf/jobs_tfidf_matrix.npz"

    # Load sparse TF-IDF matrices
    X_resumes = load_tfidf_matrix(resume_vec_path)
    X_jobs = load_tfidf_matrix(job_vec_path)

    print(f"✅ Loaded resumes vector shape: {X_resumes.shape}")
    print(f"✅ Loaded job descriptions vector shape: {X_jobs.shape}")

    # Compute cosine similarity
    similarity_matrix = compute_similarity_matrix(X_resumes, X_jobs)

    # # Uncomment to see similarity statistics like min, max, mean, median similarity scores
    # all_scores = similarity_matrix.flatten()
    # print(f"Min score: {np.min(all_scores):0.4f}, \nMax score: {np.max(all_scores):0.4f}, \nMean score: {np.mean(all_scores):0.4f}, \nMedian score: {np.median(all_scores):0.4f}")

    # Get top 5 matches per resume
    matches = top_n_tfidf_matches(similarity_matrix, top_n=5)

    # Display example output (i.e. top_n job matches for first 5 resumes)
    for resume_idx, top_jobs in list(matches.items())[:5]:
        print(f"\n🔎 Resume {resume_idx} best matches:")
        for job_idx, score in top_jobs:
            print(f"   ↳ Job {job_idx} with similarity score: {score}")
