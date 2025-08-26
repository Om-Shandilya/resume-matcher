import argparse
import os
import pandas as pd
from src.feature_engg.tfidf_vectorizing_data import load_tfidf_vectorizer, load_tfidf_matrix
from src.feature_engg.bert_embedding_data import get_bert_model, load_faiss_index
from src.processing.text_cleaning import clean_text, clean_text_for_bert
from src.matching.matching_engine import compute_similarity_matrix, top_n_tfidf_matches, top_n_bert_matches
from src.utils.file_reader import extract_text_from_file


def load_job_titles(job_csv_path: str):
    df = pd.read_csv(job_csv_path)
    if "title" not in df.columns:
        raise ValueError("Job CSV must contain a 'title' column.")
    return df


# ------------------------- TF-IDF PIPELINE -------------------------
def run_tfidf_pipeline(args, raw_resume: str):

    # Step 1: Clean resume
    cleaned_resume = clean_text(raw_resume)

    # Step 2: Load vectorizer + job matrix (local first, fallback HF)
    vectorizer = load_tfidf_vectorizer(
        local_vectorizer_path=args.local_vectorizer_path,
        repo_id=args.tfidf_repo_id,
        filename=args.vectorizer_filename
    )
    job_matrix = load_tfidf_matrix(
        local_matrix_path=args.local_matrix_path,
        repo_id=args.tfidf_repo_id,
        filename=args.matrix_filename
    )

    # Step 3: Vectorize resume
    resume_vector = vectorizer.transform([cleaned_resume])

    # Step 4: Compute cosine similarity
    sim_matrix = compute_similarity_matrix(resume_vector, job_matrix)

    # Step 5: Load job titles
    job_df = load_job_titles("data/app_data/tfidf_job_titles.csv")

    # Step 6: Get top-N job matches
    top_k = args.top_k 

    if args.top_k > len(job_df['title'].unique()):
        print(f"⚠️ Requested top_k={args.top_k} exceeds unique job titles={len(job_df['title'].unique())}. Reducing top_k.")
        top_k = len(job_df['title'].unique())

    elif args.top_k is None:
        top_k = len(job_df['title'].unique())
        print(f"\nℹ️ Showing all {top_k} job titles.\n")

    matches = top_n_tfidf_matches(sim_matrix, top_n=top_k, job_df=job_df)

    print(f"\n🎯 Top {top_k} Job Matches for the Resume (TF-IDF):")
    for job_idx, score in matches[0]:
        print(f"🔹 {job_df.iloc[job_idx]['title']} (score: {score:0.4f})")

    if args.debug:
        print("\n================ DEBUG MODE ================")
        print("\n📄--- [DEBUG - TFIDF] Cleaned Resume Preview:\n", cleaned_resume[:1000], "---")
        print(f"\n--- [DEBUG - TFIDF] Raw Similarity Scores (top {top_k}) ---")
        for job_idx, score in matches[0]:
            print(f"[{job_idx}] {job_df.iloc[job_idx]['title']} → {score:0.6f}")
        print("==============================================")


# ------------------------- BERT PIPELINE -------------------------
def run_bert_pipeline(args, raw_resume: str):

    # Step 1: Load fine-tuned ST model (local or HF Hub)
    model = get_bert_model(args.local_bert_path or args.bert_repo_id)

    # Step 2: Load FAISS index (local or HF Hub)
    job_index = load_faiss_index(
        local_index_path=args.local_index_path,
        repo_id=args.bert_repo_id,
        filename=args.index_filename
    )

    # Step 3: Clean resume text for transformer
    cleaned_resume = clean_text_for_bert(raw_resume)

    # Step 4: Embed
    resume_embedding = model.encode(
        [cleaned_resume],
        normalize_embeddings=True
    )

    # Step 5: Search
    n_jobs = job_index.ntotal
    D, I = job_index.search(resume_embedding, n_jobs)

    # Step 6: Load job titles
    job_df = load_job_titles("data/app_data/bert_job_titles.csv")

    # Step 7: Rank top-N
    top_k = args.top_k 

    if args.top_k > len(job_df['title'].unique()):
        print(f"⚠️ Requested top_k={args.top_k} exceeds unique job titles={len(job_df['title'].unique())}. Reducing top_k.")
        top_k = len(job_df['title'].unique())

    elif args.top_k is None:
        top_k = len(job_df['title'].unique())
        print(f"\nℹ️ Showing all {top_k} job titles.\n")
        
    matches = top_n_bert_matches(I, D, job_df, top_n=top_k)

    print(f"\n🎯 Top {top_k} Job Matches for the Resume (BERT):")
    for idx, score in matches:
        print(f"🔹 {job_df.iloc[idx]['title']} (score: {score:0.4f})")

    if args.debug:
        print("\n================ DEBUG MODE ================")
        print(f"\n--- [DEBUG - BERT/FAISS] Raw Similarity Scores (top {top_k}) ---")
        for idx, score in matches:
            print(f"[{idx}] {job_df.iloc[idx]['title']} → {score:0.6f}")
        print("==============================================")


# ------------------------- MAIN -------------------------
def main(args):
    try:
        if not os.path.exists(args.resume_path):
            raise FileNotFoundError(f"⚠️ Resume file not found at: {args.resume_path}")
        
        raw_resume = extract_text_from_file(args.resume_path)
        print(f"\n📄 Resume: {args.resume_path}")

        # Pipeline selector
        print(f"⚙️ Using model: {args.model.upper()}")
        if args.model == "bert":
            run_bert_pipeline(args, raw_resume)
        else:
            run_tfidf_pipeline(args, raw_resume)

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match a resume to top relevant job titles")

    # Shared args
    parser.add_argument("--resume_path", type=str, required=True, help="Path to resume file")
    parser.add_argument("--model", type=str, choices=["tfidf", "bert"], default="tfidf")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Number of top matches to return if not specified, returns all")
    parser.add_argument("--debug", action="store_true",
                        help="print raw similarity scores for both and cleaned resume for tfidf pipeline")

    # TF-IDF args
    parser.add_argument("--local_vectorizer_path", type=str, default=None,
                        help="Local TF-IDF vectorizer .pkl file")
    parser.add_argument("--local_matrix_path", type=str, default=None,
                        help="Local TF-IDF job matrix .npz file") 
    parser.add_argument("--tfidf_repo_id", type=str, default="Om-Shandilya/resume-matcher-tfidf",
                        help="Hub repo id for HuggingFace model")
    parser.add_argument("--vectorizer_filename", type=str, default="applicant/job_vectorizer.pkl",
                        help="Filename of vectorizer in the HF repo")
    parser.add_argument("--matrix_filename", type=str, default="applicant/job_matrix.npz",
                        help="Filename of matrix in the HF repo")

    # BERT args
    parser.add_argument("--local_bert_path", type=str, default=None,
                            help="Local fine-tuned ST model path")
    parser.add_argument("--local_index_path", type=str, default=None,
                            help="Local FAISS index file path")
    parser.add_argument("--bert_repo_id", type=str, default="Om-Shandilya/resume-matcher-bert",
                        help="fine-tuned ST model's HF repo id")
    parser.add_argument("--index_filename", type=str, default="applicant/jobs.faiss",
                        help="Filename of FAISS index in the HF repo")

    args = parser.parse_args()
    main(args)
