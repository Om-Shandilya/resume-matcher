import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.feature_engg.tfidf_vectorizing_data import load_tfidf_vectorizer
from src.feature_engg.bert_embedding_data import load_bert_model
from src.processing.text_cleaning import clean_text, clean_text_for_bert


def rank_with_tfidf(raw_job_text, raw_resume_texts, *,
                    local_vectorizer_path=None,
                    repo_id="Om-Shandilya/resume-matcher-tfidf",
                    filename="recruiter/combined_vectorizer.pkl",
                    top_k=None,
                    debug=False):
    """Rank resumes using TF-IDF similarity."""
    vectorizer = load_tfidf_vectorizer(
        local_vectorizer_path=local_vectorizer_path,
        repo_id=repo_id,
        filename=filename
    )

    cleaned_job_text = clean_text(raw_job_text)
    job_vector = vectorizer.transform([cleaned_job_text])

    cleaned_resumes = {fname: clean_text(txt) for fname, txt in raw_resume_texts.items()}
    resume_matrix = vectorizer.transform(cleaned_resumes.values())

    sims = cosine_similarity(job_vector, resume_matrix)[0]
    ranked = sorted(zip(cleaned_resumes.keys(), sims), key=lambda x: x[1], reverse=True)

    available_resumes = len(ranked)

    message = ""
    if top_k is None:
        final_top_k = available_resumes
        message = f"✅ Showing all {available_resumes} job matches, ranked by relevance."
    elif top_k > available_resumes:
        final_top_k = available_resumes
        message = f"ℹ️ You requested {top_k} matches, but only {available_resumes} are available. Showing all {available_resumes} matches."
    else:
        final_top_k = top_k
        message = f"✅ Showing the top {final_top_k} job matches."

    if debug:
        print("\n================ DEBUG MODE ================")
        print("\n📄--- [DEBUG - TFIDF] Cleaned Job Description Preview:\n", cleaned_job_text[:1000], "---")
        print("\n--- [DEBUG - TFIDF] First 3 Cleaned Resumes ---")
        for i, (fname, txt) in enumerate(cleaned_resumes.items()):
            if i >= 3: break
            print(f"{fname}: {txt[:300]}...\n")
        print("\n--- [DEBUG - TFIDF] Raw Similarity Scores ---")
        for fname, score in ranked[:final_top_k]:
            print(f"{fname} → {score:0.6f}")
        print("==============================================")

    return [(fname, score) for fname, score in ranked[:final_top_k]], message


def rank_with_bert(raw_job_text, raw_resume_texts, *,
                   local_bert_path=None,
                   repo_id="Om-Shandilya/resume-matcher-bert",
                   top_k=None,
                   debug=False):
    """Rank resumes using BERT embeddings."""
    model = load_bert_model(local_bert_path=local_bert_path, repo_id=repo_id)

    cleaned_job_text = clean_text_for_bert(raw_job_text)
    job_embedding = model.encode([cleaned_job_text], normalize_embeddings=True)

    cleaned_resumes = {fname: clean_text_for_bert(txt) for fname, txt in raw_resume_texts.items()}
    resume_embeddings = model.encode(list(cleaned_resumes.values()), normalize_embeddings=True)

    sims = np.dot(resume_embeddings, job_embedding.T).flatten()
    ranked = sorted(zip(cleaned_resumes.keys(), sims), key=lambda x: x[1], reverse=True)

    available_resumes = len(ranked)

    message = ""
    if top_k is None:
        final_top_k = available_resumes
        message = f"✅ Showing all {available_resumes} job matches, ranked by relevance."
    elif top_k > available_resumes:
        final_top_k = available_resumes
        message = f"ℹ️  You requested {top_k} matches, but only {available_resumes} are available. Showing all {available_resumes} matches."
    else:
        final_top_k = top_k
        message = f"✅ Showing the top {final_top_k} job matches."

    if debug:
        print("\n================ DEBUG MODE ================")
        print("\n📄--- [DEBUG - BERT] Cleaned Job Description Preview:\n", cleaned_job_text[:1000], "---")
        print("\n--- [DEBUG - BERT] First 3 Cleaned Resumes ---")
        for i, (fname, txt) in enumerate(cleaned_resumes.items()):
            if i >= 3: break
            print(f"{fname}: {txt[:300]}...\n")
        print("\n--- [DEBUG - BERT] Raw Similarity Scores ---")
        for fname, score in ranked[:final_top_k]:
            print(f"{fname} → {score:0.6f}")
        print("==============================================")
    
    return [(fname, score) for fname, score in ranked[:final_top_k]], message
