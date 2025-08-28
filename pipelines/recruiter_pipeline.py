import argparse
import os
from src.utils.bulk_loading import bulk_load_raw_resume_files
from src.utils.file_reader import extract_text_from_file
from pipelines.core import recruiter


def main(args):
    try:
        if not os.path.exists(args.job_desc_path):
            raise FileNotFoundError(f"⚠️ Job description not found: {args.job_desc_path}")
        
        raw_job_text = extract_text_from_file(args.job_desc_path)

        if not os.path.exists(args.resume_dir):
            raise FileNotFoundError(f"⚠️ Resume directory not found: {args.resume_dir}")
        
        raw_resume_texts = bulk_load_raw_resume_files(args.resume_dir)

        if not raw_resume_texts:
            raise ValueError("⚠️ No valid resumes found in the given directory.")

        print(f"\n📄 Loaded Job Description: {args.job_desc_path}")
        print(f"📂 Loaded {len(raw_resume_texts)} resumes from {args.resume_dir}")
        print(f"⚙️  Using model: {args.model.upper()}")

        if args.model == "bert":
            matches, message = recruiter.rank_with_bert(raw_job_text,
                                                         raw_resume_texts,
                                                         local_bert_path=args.local_bert_path,
                                                         repo_id=args.bert_repo_id,
                                                         top_k=args.top_k,
                                                         debug=args.debug
                                                         )
        else:
            matches, message = recruiter.rank_with_tfidf(raw_job_text,
                                                         raw_resume_texts,
                                                         local_vectorizer_path=args.local_vectorizer_path,
                                                         repo_id=args.tfidf_repo_id,
                                                         filename=args.vectorizer_filename,
                                                         top_k=args.top_k,
                                                         debug=args.debug
                                                         )
        
        print(f"\n{message}")
        print(f"\n🎯 Top {len(matches)} Job Matches ({args.model.upper()}):")
        for i, (job, score) in enumerate(matches):
            print(f"{i+1})-> {job} (score: {score:.4f})")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recruiter Pipeline: Rank resumes for a given job description")

    parser.add_argument("--job_desc_path", type=str, required=True)
    parser.add_argument("--resume_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["tfidf", "bert"], default="tfidf")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    # TF-IDF args
    parser.add_argument("--local_vectorizer_path", type=str, default=None)
    parser.add_argument("--tfidf_repo_id", type=str, default="Om-Shandilya/resume-matcher-tfidf")
    parser.add_argument("--vectorizer_filename", type=str, default="recruiter/combined_vectorizer.pkl")

    # BERT args
    parser.add_argument("--local_bert_path", type=str, default=None)
    parser.add_argument("--bert_repo_id", type=str, default="Om-Shandilya/resume-matcher-bert")

    args = parser.parse_args()
    main(args)
