import argparse
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.feature_engg.tfidf_vectorizing_data import load_tfidf_vectorizer
from src.feature_engg.bert_embedding_data import get_bert_model
from src.utils.bulk_loading import bulk_load_raw_resume_files
from src.utils.file_reader import extract_text_from_file
from src.processing.text_cleaning import clean_text, clean_text_for_bert


# ------------------------- TF-IDF PIPELINE -------------------------
def run_tfidf_pipeline(args, raw_job_text, raw_resume_texts):
    # Step 1: Load vectorizer (local or HF Hub)
    vectorizer = load_tfidf_vectorizer(
        local_vectorizer_path=args.local_vectorizer_path,
        repo_id=args.tfidf_repo_id,
        filename=args.vectorizer_filename
    )

    # Step 2: Clean job description
    cleaned_job_text = clean_text(raw_job_text)
    job_vector = vectorizer.transform([cleaned_job_text])

    # Step 3: Clean and vectorize resumes
    cleaned_resumes = {fname: clean_text(txt) for fname, txt in raw_resume_texts.items()}
    resume_matrix = vectorizer.transform(cleaned_resumes.values())

    # Step 4: Compute similarity
    sims = cosine_similarity(job_vector, resume_matrix)[0]

    # Step 5: Rank resumes
    ranked = sorted(zip(cleaned_resumes.keys(), sims), key=lambda x: x[1], reverse=True)

    # Step 6: Top-K handling
    top_k = args.top_k
    available_resumes = len(ranked)

    if args.top_k is None:
        top_k = available_resumes
        print(f"\nℹ️ Showing all {available_resumes} resumes.\n")
    elif args.top_k > available_resumes:
        top_k = available_resumes
        print(f"\n⚠️ Requested top_k={args.top_k} exceeds available resumes={available_resumes}. Reducing top_k.\n")

    print(f"\n🎯 Top {top_k} Candidate Matches for the Job (TF-IDF):")
    for i, (fname, score) in enumerate(ranked[:top_k], 1):
        print(f"{i}. {fname} → score: {score:.4f}")

    if args.debug:
        print("\n================ DEBUG MODE ================")
        print("\n📄--- [DEBUG - TFIDF] Cleaned Job Description Preview:\n", cleaned_job_text[:1000], "---")
        print("\n--- [DEBUG - TFIDF] First 3 Cleaned Resumes ---")
        for i, (fname, txt) in enumerate(cleaned_resumes.items()):
            if i >= 3: break
            print(f"{fname}: {txt[:300]}...\n")
        print(f"\n--- [DEBUG - TFIDF] Raw Similarity Scores (top {top_k}) ---")
        for fname, score in ranked[:top_k]:
            print(f"{fname} → {score:0.6f}")
        print("==============================================")


# ------------------------- BERT PIPELINE -------------------------
def run_bert_pipeline(args, raw_job_text, raw_resume_texts):
    # Step 1: Load fine-tuned ST model (local or HF Hub)
    model = get_bert_model(args.local_bert_path or args.bert_repo_id)

    # Step 2: Clean job description
    cleaned_job_text = clean_text_for_bert(raw_job_text)
    job_embedding = model.encode([cleaned_job_text], normalize_embeddings=True)

    # Step 3: Encode resumes
    cleaned_resumes = {fname: clean_text_for_bert(txt) for fname, txt in raw_resume_texts.items()}
    resume_embeddings = model.encode(list(cleaned_resumes.values()), normalize_embeddings=True)

    # Step 4: Compute cosine similarity manually
    # Using dot product as embeddings are normalized and not FAISS since we have small data here.
    sims = np.dot(resume_embeddings, job_embedding.T).flatten()

    # Step 5: Rank resumes
    ranked = sorted(zip(cleaned_resumes.keys(), sims), key=lambda x: x[1], reverse=True)

    # Step 6: Top-K handling
    top_k = args.top_k
    available_resumes = len(ranked)

    if args.top_k is None:
        top_k = available_resumes
        print(f"\nℹ️ Showing all {available_resumes} resumes.\n")
    elif args.top_k > available_resumes:
        top_k = available_resumes
        print(f"\n⚠️ Requested top_k={args.top_k} exceeds available resumes={available_resumes}. Reducing top_k.\n")

    print(f"\n🎯 Top {top_k} Candidate Matches for the Job (BERT):")
    for i, (fname, score) in enumerate(ranked[:top_k], 1):
        print(f"{i}. {fname} → score: {score:.4f}")

    if args.debug:
        print("\n================ DEBUG MODE ================")
        print("\n📄--- [DEBUG - BERT] Cleaned Job Description Preview:\n", cleaned_job_text[:1000], "---")
        print("\n--- [DEBUG - BERT] First 3 Cleaned Resumes ---")
        for i, (fname, txt) in enumerate(cleaned_resumes.items()):
            if i >= 3: break
            print(f"{fname}: {txt[:300]}...\n")
        print(f"\n--- [DEBUG - BERT] Raw Similarity Scores (top {top_k}) ---")
        for fname, score in ranked[:top_k]:
            print(f"{fname} → {score:0.6f}")
        print("==============================================")


# ------------------------- MAIN -------------------------
def main(args):
    try:
        # Load job description and resumes
        if not os.path.exists(args.job_desc_path):
            raise FileNotFoundError(f"⚠️ Job description not found: {args.job_desc_path}")
        raw_job_text = extract_text_from_file(args.job_desc_path)

        if not os.path.exists(args.resume_dir):
            raise FileNotFoundError(f"⚠️ Resume directory not found: {args.resume_dir}")
        raw_resume_texts = bulk_load_raw_resume_files(args.resume_dir)

        if not raw_resume_texts:
            raise ValueError("⚠️ No valid resumes found in the given directory.")

        print(f"\n📄 Job Description: {args.job_desc_path}")
        print(f"📂 Loaded {len(raw_resume_texts)} resumes from {args.resume_dir}")

        # Pipeline selector
        print(f"⚙️ Using model: {args.model.upper()}")
        if args.model == "bert":
            run_bert_pipeline(args, raw_job_text, raw_resume_texts)
        else:
            run_tfidf_pipeline(args, raw_job_text, raw_resume_texts)

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recruiter Pipeline: Rank resumes for a given job description")

    # Shared args
    parser.add_argument("--job_desc_path", type=str, required=True, help="Path to job description file")
    parser.add_argument("--resume_dir", type=str, required=True, help="Directory containing applicant resumes")
    parser.add_argument("--model", type=str, choices=["tfidf", "bert"], default="tfidf")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Number of top matches to return if not specified, returns all")
    parser.add_argument("--debug", action="store_true",
                        help="print raw similarity scores and cleaned texts for debugging")

    # TF-IDF args
    parser.add_argument("--local_vectorizer_path", type=str, default=None,
                        help="Local TF-IDF vectorizer .pkl file")
    parser.add_argument("--tfidf_repo_id", type=str, default="Om-Shandilya/resume-matcher-tfidf",
                        help="Hub repo id for HuggingFace TF-IDF model")
    parser.add_argument("--vectorizer_filename", type=str, default="recruiter/combined_vectorizer.pkl",
                        help="Filename of vectorizer in the HF repo")

    # BERT args
    parser.add_argument("--local_bert_path", type=str, default=None,
                        help="Local fine-tuned ST model path")
    parser.add_argument("--bert_repo_id", type=str, default="Om-Shandilya/resume-matcher-bert",
                        help="fine-tuned ST model's HF repo id")

    args = parser.parse_args()
    main(args)
