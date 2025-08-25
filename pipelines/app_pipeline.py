import argparse
import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from src.feature_engg.tfidf_vectorizing_data import load_vectorizer, load_vector_data
from src.processing.text_cleaning import clean_text
from src.matching.matching_engine import compute_similarity_matrix, top_n_tfidf_matches, top_n_bert_matches
from src.utils.file_reader import extract_text_from_file


def load_job_titles(job_csv_path: str):
    df = pd.read_csv(job_csv_path)
    if 'title' not in df.columns:
        raise ValueError("Job CSV must contain a 'title' column.")
    return df


def run_tfidf_pipeline(args, raw_resume: str):

    # Step 2: Clean resume text
    cleaned_resume = clean_text(raw_resume)

    # Step 3: Load vectorizer and job matrix
    vectorizer = load_vectorizer(args.vectorizer_path)
    job_matrix = load_vector_data(args.job_matrix_path)

    # Step 4: Vectorize cleaned resume text
    resume_vector = vectorizer.transform([cleaned_resume])

    # Step 5: Compute similarity
    sim_matrix = compute_similarity_matrix(resume_vector, job_matrix)

    # Step 6: Load job titles
    job_df = load_job_titles(args.job_title_csv)

    # Step 7: Get top-N job matches
    matches = top_n_tfidf_matches(sim_matrix, top_n=args.top_k, job_df=job_df)

    print(f"\n🎯 Top {args.top_k} Job Matches for the Resume (TF-IDF):")
    for job_idx, score in matches[0]:
        print(f"🔹 {job_df.iloc[job_idx]['title']} (score: {score:0.4f})")

    # Optional debug
    if args.debug:
        print("\n================ DEBUG MODE ================")
        print("\n📄--- [DEBUG - TFIDF] Cleaned Resume Preview:\n", cleaned_resume[:1000], "---")
        print(f"\n--- [DEBUG - TFIDF] Raw Similarity Scores (top {args.top_k}) ---")
        for job_idx, score in matches[0]:
            print(f"[{job_idx}] {job_df.iloc[job_idx]['title']} → {score:0.6f}")
        print("==============================================")


def run_bert_pipeline(args, raw_resume: str):
    # Step 2: Load SentenceTransformer model
    model = SentenceTransformer(args.bert_model_path)

    # Step 3: Load FAISS job index
    job_index = faiss.read_index(args.bert_faiss_index)

    # Step 4: Encode resume into embedding
    resume_embedding = model.encode([raw_resume], normalize_embeddings=True)

    # Step 5: Search deeply in FAISS index in order to eliminate duplicate job titles
    # Search across all job embeddings in FAISS
    n_jobs = job_index.ntotal
    D, I = job_index.search(resume_embedding, n_jobs)

    # Step 6: Load job titles
    job_df = load_job_titles(args.job_title_csv)

    print(f"\n🎯 Top {args.top_k} Job Matches for the Resume (BERT):")
    matches = top_n_bert_matches(I, D, job_df, top_n=args.top_k)

    for idx, score in matches:
        print(f"🔹 {job_df.iloc[idx]['title']} (score: {score:0.4f})")

    # Optional debug
    if args.debug:
        print("\n================ DEBUG MODE ================")
        print(f"\n--- [DEBUG - BERT/FAISS] Raw Similarity Scores (top {args.top_k}) ---")
        for idx, score in matches:
            print(f"🔹 {job_df.iloc[idx]['title']} (score: {score})")
        print("==============================================")


def main(args):
    try:
        # Step 1: Load raw resume text
        if not os.path.exists(args.resume_path):
            raise FileNotFoundError(f"Resume file not found: {args.resume_path}")
        raw_resume = extract_text_from_file(args.resume_path)

        # Run chosen pipeline
        if args.model == "bert":
            run_bert_pipeline(args, raw_resume)
        else:
            run_tfidf_pipeline(args, raw_resume)

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match a resume to top relevant job titles")
    parser.add_argument('--resume_path', type=str, required=True, help="Path to resume file")
    parser.add_argument('--model', type=str, choices=['tfidf', 'bert'], default='tfidf',
                        help="Which model pipeline to use: 'tfidf' or 'bert'")


    # TF-IDF arguments
    parser.add_argument('--vectorizer_path', type=str, default='models/tfidf/app_tfidf/job_tfidf_vectorizer.pkl')
    parser.add_argument('--job_matrix_path', type=str, default='models/tfidf/app_tfidf/job_tfidf_matrix.npz')

    # BERT arguments
    parser.add_argument('--bert_model_path', type=str, default='models/bert/dapt_minilm_sentence_transformer',
                        help="Path to fine-tuned SentenceTransformer model")
    parser.add_argument('--bert_faiss_index', type=str, default='models/bert/app_bert/jobs_bert_embeddings.faiss',
                        help="Path to FAISS index of job embeddings")

    # Shared arguments
    parser.add_argument('--job_title_csv', type=str, default='data/app_data/job_titles.csv')
    parser.add_argument('--top_k', type=int, default=5,
                        help="Number of top job matches to return")
    parser.add_argument('--debug', action='store_true',
                        help="Print raw similarity scores and cleaned resume for tfidf pipeline")

    args = parser.parse_args()
    main(args)

