import argparse, os
from src.utils.file_reader import extract_text_from_file
from pipelines.core.applicant import run_tfidf_pipeline, run_bert_pipeline

def main(args):
    try:
        if not os.path.exists(args.resume_path):
            raise FileNotFoundError(f"⚠️ Resume not found at {args.resume_path}")
        raw_resume = extract_text_from_file(args.resume_path)

        if args.model == "bert":
            matches, message = run_bert_pipeline(raw_resume,
                                        local_bert_path=args.local_bert_path,
                                        local_index_path=args.local_index_path,
                                        repo_id=args.bert_repo_id,
                                        index_filename=args.index_filename,
                                        top_k=args.top_k,
                                        debug=args.debug)
        else:
            matches, message = run_tfidf_pipeline(raw_resume,
                                        local_vectorizer_path=args.local_vectorizer_path,
                                        local_matrix_path=args.local_matrix_path,
                                        repo_id=args.tfidf_repo_id,
                                        vectorizer_filename=args.vectorizer_filename,
                                        matrix_filename=args.matrix_filename,
                                        top_k=args.top_k,
                                        debug=args.debug)
            
        print(f"\n{message}")
        print(f"\n🎯 Top {len(matches)} Job Matches ({args.model.upper()}):")
        for fname, score in matches:
            print(f"🔹 {fname} (score: {score:.4f})")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match a resume to top relevant job titles")
    parser.add_argument("--resume_path", type=str, required=True)
    parser.add_argument("--model", choices=["tfidf","bert"], default="tfidf")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="print raw similarity scores for both and cleaned resume for tfidf pipeline")

    # tfidf args
    parser.add_argument("--local_vectorizer_path", type=str, default=None)
    parser.add_argument("--local_matrix_path", type=str, default=None)
    parser.add_argument("--tfidf_repo_id", type=str, default="Om-Shandilya/resume-matcher-tfidf")
    parser.add_argument("--vectorizer_filename", type=str, default="applicant/job_vectorizer.pkl")
    parser.add_argument("--matrix_filename", type=str, default="applicant/job_matrix.npz")

    # bert args
    parser.add_argument("--local_bert_path", type=str, default=None)
    parser.add_argument("--local_index_path", type=str, default=None)
    parser.add_argument("--bert_repo_id", type=str, default="Om-Shandilya/resume-matcher-bert")
    parser.add_argument("--index_filename", type=str, default="applicant/jobs.faiss")

    args = parser.parse_args()
    main(args)
