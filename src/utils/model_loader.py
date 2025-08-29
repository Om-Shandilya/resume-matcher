import streamlit as st
from src.feature_engg.bert_embedding_data import load_bert_model
from src.feature_engg.tfidf_vectorizing_data import load_tfidf_vectorizer, load_tfidf_matrix
from src.feature_engg.bert_embedding_data import load_faiss_index

# Usinf streamlit's caching mechanism to load models and artifacts only once
@st.cache_resource
def get_bert_model():
    """Loads and caches the BERT model."""

    return load_bert_model(local_bert_path=None,
                           repo_id="Om-Shandilya/resume-matcher-bert")

@st.cache_resource
def get_faiss_index():
    """Loads and caches the FAISS index for the applicant view."""

    return load_faiss_index(local_index_path=None,
                            repo_id="Om-Shandilya/resume-matcher-bert", 
                            filename="applicant/jobs.faiss")

@st.cache_resource
def get_applicant_vectorizer():
    """Loads and caches the TF-IDF vectorizer for the applicant view."""
    
    return load_tfidf_vectorizer(local_vectorizer_path=None,
                                 repo_id="Om-Shandilya/resume-matcher-tfidf", 
                                 filename="applicant/job_vectorizer.pkl")
@st.cache_resource
def get_applicant_matrix():
    """Loads and caches the TF-IDF matrix for the applicant view."""

    return load_tfidf_matrix(local_matrix_path=None,
                             repo_id="Om-Shandilya/resume-matcher-tfidf", 
                             filename="applicant/job_matrix.npz")
    

@st.cache_resource
def get_recruiter_vectorizer():
    """Loads and caches the TF-IDF vectorizer for the recruiter view."""

    return load_tfidf_vectorizer(local_vectorizer_path=None,
                                 repo_id="Om-Shandilya/resume-matcher-tfidf", 
                                 filename="recruiter/combined_vectorizer.pkl")