import argparse
import os
import pandas as pd
from src.feature_engg.vectorizing_data import load_vectorizer, load_vector_data, vectorize_text
from src.processing.text_cleaning import clean_text
from src.matching.matching_engine import compute_similarity_matrix, top_n_matches
from src.utils.file_reader import extract_text_from_file


def load_job_titles(job_csv_path: str):
    df = pd.read_csv(job_csv_path)
    if 'title' not in df.columns:
        raise ValueError("Job CSV must contain a 'title' column.")
    return df


def main(args):
    try:
        # Step 1: Load and clean resume text (supports .pdf, .docx, .txt)
        if not os.path.exists(args.resume_path):
            raise FileNotFoundError(f"Resume file not found: {args.resume_path}")
        raw_resume = extract_text_from_file(args.resume_path)
        cleaned_resume = clean_text(raw_resume)


        # Step 2: Load vectorizer and job matrix
        vectorizer = load_vectorizer(args.vectorizer_path)
        job_matrix = load_vector_data(args.job_matrix_path)

        # Step 3: Vectorize cleaned resume text
        resume_vector = vectorizer.transform([cleaned_resume])  # single row sparse matrix

        # Step 4: Compute similarity
        sim_matrix = compute_similarity_matrix(resume_vector, job_matrix)

        # Step 5: Load job titles for display
        job_df = load_job_titles(args.job_title_csv)

        # Step 6: Get top-N job matches
        matches = top_n_matches(sim_matrix, top_n=args.top_k, job_df=job_df)

        print(f"\n🎯 Top {args.top_k} Job Matches for the Resume:")
        for job_idx, score in matches[0]:  # 0 because it's the only resume
            print(f"🔹 {job_df.iloc[job_idx]['title']} (score: {score:0.4f})")

        # Optional debug output
        if args.debug:
            print("\n===== DEBUG MODE =====")
            print("\n📄 Cleaned Resume Preview:\n", cleaned_resume[:1000])
            print("\n📊 Raw Similarity Scores:\n", sim_matrix)
            print("=======================")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match a resume to top relevant job titles")
    parser.add_argument('--resume_path', type=str, required=True, help="Path to resume file")
    parser.add_argument('--vectorizer_path', type=str, default='models/app_tfidf/job_tfidf_vectorizer.pkl')
    parser.add_argument('--job_matrix_path', type=str, default='models/app_tfidf/job_tfidf_matrix.npz')
    parser.add_argument('--job_title_csv', type=str, default='data/app_data/job_titles.csv')
    parser.add_argument('--top_k', type=int, default=5, help="Number of top job matches to return")
    parser.add_argument('--debug', action='store_true', help="Print cleaned resume and raw matches")

    args = parser.parse_args()
    main(args)

