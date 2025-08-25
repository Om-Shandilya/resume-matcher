import argparse
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sentence_transformers import SentenceTransformer
from src.utils.bulk_loading import bulk_load_raw_resume_files
from src.utils.file_reader import extract_text_from_file
from src.processing.text_cleaning import clean_text


def rank_with_tfidf(args, raw_job_text, raw_resume_texts):
    """TF-IDF recruiter pipeline"""
    # Step 1: Load vectorizer
    if not os.path.exists(args.vectorizer_path):
        raise FileNotFoundError(f"⚠️ Vectorizer file not found: {args.vectorizer_path}")
    vectorizer = joblib.load(args.vectorizer_path)

    # Step 2: Process job description
    cleaned_job_text = clean_text(raw_job_text)
    job_vector = vectorizer.transform([cleaned_job_text])

    # Step 3: Process resumes
    cleaned_resumes = {fname: clean_text(txt) for fname, txt in raw_resume_texts.items()}
    resume_matrix = vectorizer.transform(cleaned_resumes.values())

    # Step 4: Compute similarity
    sims = cosine_similarity(job_vector, resume_matrix)[0]

    if args.debug:
        print("\n================ DEBUG MODE ================")
        print("\n[DEBUG - TFIDF] Cleaned job description:")
        print(cleaned_job_text[:500], "...\n")
        print("[DEBUG - TFIDF] First 3 cleaned resumes:")
        for i, (fname, txt) in enumerate(cleaned_resumes.items()):
            if i >= 3: break
            print(f"{fname}: {txt[:300]}...\n")
        print("[DEBUG - TFIDF] Raw similarity scores:", sims[:10])
        print("==============================================")

    # Step 5: Rank resumes
    ranked = sorted(zip(cleaned_resumes.keys(), sims), key=lambda x: x[1], reverse=True)
    return ranked


def rank_with_bert(args, raw_job_text, raw_resume_texts):
    """BERT recruiter pipeline using FAISS (on the fly)"""
    if not os.path.exists(args.bert_model_path):
        raise FileNotFoundError(f"⚠️ BERT model not found: {args.bert_model_path}")

    # Step 1: Load BERT model
    model = SentenceTransformer(args.bert_model_path)

    # Step 2: Encode job description
    job_embedding = model.encode([raw_job_text], convert_to_numpy=True, normalize_embeddings=True)

    # Step 3: Encode resumes
    resume_embeddings = model.encode(list(raw_resume_texts.values()), convert_to_numpy=True, normalize_embeddings=True)

    # Step 4: Create FAISS indices
    local_index = faiss.IndexFlatIP(resume_embeddings.shape[1])
    local_index.add(resume_embeddings)

    scores, indices = local_index.search(job_embedding, len(raw_resume_texts))

    if args.debug:
        print("\n================ DEBUG MODE ================")
        print("\n[DEBUG - BERT/FAISS] Raw job description:")
        print(raw_job_text[:500], "...\n")
        print("[DEBUG - BERT/FAISS] First 3 raw resumes:")
        for i, (fname, txt) in enumerate(raw_resume_texts.items()):
            if i >= 3: break
            print(f"{fname}: {txt[:300]}...\n")
        print(f"[DEBUG - BERT/FAISS] all similarity scores:", scores[0][:len(raw_resume_texts)])
        print("==============================================")

    # Step 5: Rank resumes
    ranked = [(list(raw_resume_texts.keys())[i], float(scores[0][j]))
              for j, i in enumerate(indices[0])]
    return ranked


def main(args):
    try:
        # Load raw job and resumes 
        raw_job_text = extract_text_from_file(args.job_desc_path)
        raw_resume_texts = bulk_load_raw_resume_files(args.resume_dir)

        if not raw_resume_texts:
            raise ValueError("⚠️ No valid resumes found in the given directory.")

        # Limit the number of resumes displayed based on the top_k argument and available resumes
        available_resumes = len(raw_resume_texts)
        top_k = min(args.top_k, available_resumes)

        if args.top_k > available_resumes:
            print(f"\n⚠️ Only {available_resumes} resumes are available. "
                  f"Showing top {available_resumes} matches instead of {args.top_k}.\n")

        # Choose model
        if args.model == "tfidf":
            ranked = rank_with_tfidf(args, raw_job_text, raw_resume_texts)
        elif args.model == "bert":
            ranked = rank_with_bert(args, raw_job_text, raw_resume_texts)
        else:
            raise ValueError("❌ Invalid model. Choose 'tfidf' or 'bert'.")

        # Display ranked resumes
        print(f"\n🎯 Top {top_k} Candidate Matches for the Job ({args.model.upper()}):")
        for i, (fname, score) in enumerate(ranked[:top_k], 1):
            print(f"{i}. {fname}  → score: {score:.4f}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recruiter Pipeline: Rank user uploaded resumes for a given job description")

    # Shared arguments
    parser.add_argument('--job_desc_path', type=str, required=True, help="Path to job description file")
    parser.add_argument('--resume_dir', type=str, required=True, help="Directory containing applicant resumes")
    parser.add_argument('--model', type=str, choices=['tfidf', 'bert'], default='tfidf',
                        help="Model to use: tfidf or bert")
    parser.add_argument('--top_k', type=int, default=10, help="Number of top resumes to return")
    parser.add_argument('--debug', action='store_true', help="Print cleaned/raw texts and raw similarity scores")

    # TF-IDF specific
    parser.add_argument('--vectorizer_path', type=str,
                        default='models/tfidf/recruiter_tfidf/combined_tfidf_vectorizer.pkl',
                        help="Path to pre-trained TF-IDF vectorizer")

    # BERT specific
    parser.add_argument('--bert_model_path', type=str,
                        default='models/bert/dapt_minilm_sentence_transformer',
                        help="Path to fine-tuned BERT/SBERT model")

    args = parser.parse_args()
    main(args)
