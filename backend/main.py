from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
import pandas as pd
import sys
import os
import logging

# This ensures that the backend can find your 'src' and 'pipelines' modules and also adds the parent directory to sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.models import (ResumeRequest, ApplicantResponse, JobMatch,
                            RecruiterRequest, RecruiterResponse, ResumeMatch)
from pipelines.core.applicant import  run_bert_pipeline, run_tfidf_pipeline, load_job_titles
from pipelines.core.recruiter import  rank_with_bert, rank_with_tfidf
from src.feature_engg.bert_embedding_data import load_bert_model, load_faiss_index
from src.feature_engg.tfidf_vectorizing_data import load_tfidf_vectorizer, load_tfidf_matrix

# In memory storage for models (dictionary to hold all loaded models):
ml_models = {}

# Create a lifespan function to handle startup and shutdown events:
@asynccontextmanager
async def lifespan(app: FastAPI):
    """This code runs ONCE when the server starts up."""
    print("🚀 Server starting up: Loading all ML models and data from the Hub...")
    
    # Loading BERT Models
    ml_models["bert_model"] = load_bert_model(repo_id="Om-Shandilya/resume-matcher-bert", local_bert_path=None)
    ml_models["faiss_index"] = load_faiss_index(repo_id="Om-Shandilya/resume-matcher-bert", filename="applicant/jobs.faiss", lazy_loading=True, local_index_path=None)
    ml_models["bert_job_df"] = load_job_titles(repo_id='Om-Shandilya/resume-matcher-bert', filename='applicant/bert_job_titles.csv', local_path=None)

    # Loading TF-IDF Models
    print("📂 Loading pre-computed TF-IDF models...")
    ml_models["applicant_vectorizer"] = load_tfidf_vectorizer(repo_id="Om-Shandilya/resume-matcher-tfidf", filename="applicant/job_vectorizer.pkl", local_vectorizer_path=None)
    ml_models["applicant_matrix"] = load_tfidf_matrix(repo_id="Om-Shandilya/resume-matcher-tfidf", filename="applicant/job_matrix.npz", local_matrix_path=None)
    ml_models["recruiter_vectorizer"] = load_tfidf_vectorizer(repo_id="Om-Shandilya/resume-matcher-tfidf", filename="recruiter/combined_vectorizer.pkl", local_vectorizer_path=None)
    ml_models["tfidf_job_df"] = load_job_titles(repo_id='Om-Shandilya/resume-matcher-tfidf', filename='applicant/tfidf_job_titles.csv', local_path=None)
    
    print("✅ All models and data loaded successfully.")
    yield
    print(" shutting down: Clearing models...")
    ml_models.clear()

# Initializing the FastAPI app
app = FastAPI(
    title="Resume-Job Matcher API",
    description="An API for matching resumes to jobs and ranking candidates.",
    lifespan=lifespan
)

# Creating the API endpoints:
@app.get("/")
def read_root():
    return {"status": "Resume Matcher API is running."}

# Applicant side endpoints:
@app.post("/applicant/match/bert", response_model=ApplicantResponse)
async def match_resume_bert(request: ResumeRequest):
    try:
        matches, message = run_bert_pipeline(
            raw_resume=request.raw_text,
            model=ml_models["bert_model"],
            job_index=ml_models["faiss_index"],
            job_df=ml_models["bert_job_df"],
            top_k=request.top_k)
        
        response_matches = [JobMatch(job_title=title, match_score=score) for title, score in matches]
        return ApplicantResponse(matches=response_matches, message=message)
    except Exception as e:
        logging.exception("Error in /applicant/match/bert endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/applicant/match/tf-idf", response_model=ApplicantResponse)
async def match_resume_tfidf(request: ResumeRequest):
    try:
        matches, message = run_tfidf_pipeline(
            raw_resume=request.raw_text,
            vectorizer=ml_models["applicant_vectorizer"],
            job_matrix=ml_models["applicant_matrix"],
            job_df=ml_models["tfidf_job_df"],
            top_k=request.top_k)
        
        response_matches = [JobMatch(job_title=title, match_score=score) for title, score in matches]
        return ApplicantResponse(matches=response_matches, message=message)
    except Exception as e:
        logging.exception("Error in /applicant/match/tf-idf endpoint")
        raise HTTPException(status_code=500, detail=str(e))
        

# Recruiter side endpoints:
@app.post("/recruiter/rank/bert", response_model=RecruiterResponse)
async def rank_resumes_bert(request: RecruiterRequest):
    try:
        matches, message = rank_with_bert(
            raw_job_text=request.raw_job_text,
            raw_resume_texts=request.raw_resume_texts,
            model=ml_models["bert_model"],
            top_k=request.top_k)
        
        response_matches = [ResumeMatch(resume_filename=fname, match_score=score) for fname, score in matches]
        return RecruiterResponse(matches=response_matches, message=message)
    except Exception as e:
        logging.exception("Error in /recruiter/rank/bert endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recruiter/rank/tf-idf", response_model=RecruiterResponse)
async def rank_resumes_tfidf(request: RecruiterRequest):
    try:
        matches, message = rank_with_tfidf(
            raw_job_text=request.raw_job_text,
            raw_resume_texts=request.raw_resume_texts,
            vectorizer=ml_models["recruiter_vectorizer"],
            top_k=request.top_k)
        
        response_matches = [ResumeMatch(resume_filename=fname, match_score=score) for fname, score in matches]
        return RecruiterResponse(matches=response_matches, message=message)
    except Exception as e:
        logging.exception("Error in /recruiter/rank/tf-idf endpoint")
        raise HTTPException(status_code=500, detail=str(e))