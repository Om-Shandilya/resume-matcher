import argparse
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.bulk_loading import bulk_load_cleaned_resume_files
from src.utils.file_reader import extract_text_from_file
from src.processing.text_cleaning import clean_text


def main(args):
    try:
        # Step 1: Load vectorizer
        if not os.path.exists(args.vectorizer_path):
            raise FileNotFoundError(f"⚠️ Vectorizer file not found: {args.vectorizer_path}")
        vectorizer = joblib.load(args.vectorizer_path)


        # Step 2: Process job description
        if not os.path.exists(args.job_desc_path):
            raise FileNotFoundError(f"⚠️ Job description file not found: {args.job_desc_path}")
        
        raw_job_text = extract_text_from_file(args.job_desc_path)
        cleaned_job_text = clean_text(raw_job_text)
        job_vector = vectorizer.transform([cleaned_job_text])


        # Step 3: Process applicant resumes
        if not os.path.isdir(args.resume_dir):
            raise NotADirectoryError(f"⚠️ Resume directory not found: {args.resume_dir}")
        
        resume_texts = bulk_load_cleaned_resume_files(args.resume_dir)  # dict: {filename: cleaned_text}
        
        if not resume_texts:
            raise ValueError("⚠️ No valid resumes found in the given directory.")

        resume_matrix = vectorizer.transform(resume_texts.values())


        # Step 4: Compute similarity
        sims = cosine_similarity(job_vector, resume_matrix)[0]


        # Step 5: Rank resumes
        ranked = sorted(zip(resume_texts.keys(), sims), key=lambda x: x[1], reverse=True)
        top_k = min(args.top_k, len(ranked))


        # Step 6: Print the output
        print(f"\n🎯 Top {top_k} Candidate Matches for the Job:")
        for i, (fname, score) in enumerate(ranked[:top_k], 1):
            print(f"{i}. {fname}  → score: {score:.4f}")

        # Optional debug output
        if args.debug:
            print("\n===== DEBUG MODE =====")
            print("\n📄 Cleaned Job Description Preview:\n", cleaned_job_text[:1000])
            print("\n📊 Raw Similarity Scores:\n", sims)
            print("=======================")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recruiter Pipeline: Rank resumes for a given job description")
    parser.add_argument('--job_desc_path', type=str, required=True, help="Path to job description file")
    parser.add_argument('--resume_dir', type=str, required=True, help="Directory containing applicant resumes")
    parser.add_argument('--vectorizer_path', type=str, default='models/recruiter_tfidf/combined_tfidf_vectorizer.pkl')
    parser.add_argument('--top_k', type=int, default=10, help="Number of top resumes to return")
    parser.add_argument('--debug', action='store_true', help="Print cleaned job/resume text and raw matches")

    args = parser.parse_args()
    main(args)
