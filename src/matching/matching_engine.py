import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.feature_engg.vectorizing_data import load_vector_data

def compute_similarity_matrix(X_resumes, X_jobs ):
    """
    Compute cosine similarity between resume vectors and job vectors.
    Returns a similarity matrix of shape (num_resumes, num_jobs).
    """
    return cosine_similarity(X_resumes, X_jobs)

def top_n_matches(similarity_matrix: np.ndarray, 
                  top_n: int = 5):
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
        top_indices = row.argsort()[::-1][:top_n]
        results[i] = [(j, round(row[j], 4)) for j in top_indices]
    return results

if __name__ == "__main__":
    # Define paths
    resume_vec_path = "models/tfidf/resumes_tfidf_matrix.npz"
    job_vec_path = "models/tfidf/jobs_tfidf_matrix.npz"

    # Load sparse TF-IDF matrices
    X_resumes = load_vector_data(resume_vec_path)
    X_jobs = load_vector_data(job_vec_path)

    print(f"✅ Loaded resumes vector shape: {X_resumes.shape}")
    print(f"✅ Loaded job descriptions vector shape: {X_jobs.shape}")

    # Compute cosine similarity
    similarity_matrix = compute_similarity_matrix(X_resumes, X_jobs)

    # # Uncomment to see similarity statistics like min, max, mean, median similarity scores
    # all_scores = similarity_matrix.flatten()
    # print(f"Min score: {np.min(all_scores):0.4f}, \nMax score: {np.max(all_scores):0.4f}, \nMean score: {np.mean(all_scores):0.4f}, \nMedian score: {np.median(all_scores):0.4f}")

    # Get top 5 matches per resume
    matches = top_n_matches(similarity_matrix, top_n=5)

    # Display example output (i.e. top_n job matches for first 5 resumes)
    for resume_idx, top_jobs in list(matches.items())[:5]:
        print(f"\n🔎 Resume {resume_idx} best matches:")
        for job_idx, score in top_jobs:
            print(f"   ↳ Job {job_idx} with similarity score: {score}")
