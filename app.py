import streamlit as st
import os
import sys
import tempfile
import pandas as pd
import shutil
import altair as alt
import requests # Import for making API requests
from src.utils.bulk_loading import bulk_load_raw_resume_files
from src.utils.file_reader import extract_text_from_file

# Configuring the backend API URL
API_URL = "https://om-shandilya-resume-matcher-api.hf.space"

# Configuring the Streamlit app
st.set_page_config(
    page_title="Resume-Job Matcher",
    page_icon="🎯",
    layout="wide"
)

# Main app title and description
st.title("🎯 AI-Powered Resume-Job Matcher")
st.write("---")

# Creating sidebar for controls
with st.sidebar:
    st.header("Controls")
    app_mode = st.radio(
        "Choose your view",
        ("Applicant", "Recruiter"),
        help="Select 'Applicant' to match your resume to jobs titles. Select 'Recruiter' to rank resumes for a job."
    )
    model_choice = st.selectbox(
        "Choose the AI Model",
        ("TF-IDF", "BERT"),
        help="TF-IDF is baseline. BERT is more accurate and semantic."
    )
    st.write("---")
    show_all = st.checkbox("Show all matches", value=False)
    if show_all:
        top_k = None
        st.slider(
            "Number of matches to show",
            min_value=1, max_value=50, value=5, step=1,
            disabled=True
        )
        st.info("Showing all ranked results.")
    else:
        top_k = st.slider(
            "Number of matches to show",
            min_value=1, max_value=50, value=5, step=1,
            disabled=False
        )

# Applicant view of the app
if app_mode == "Applicant":
    st.header("Applicant: Match Your Resume to a Job")
    resume_file = st.file_uploader(
        "Upload your resume",
        type=['pdf', 'docx', 'txt'],
        help="Please upload your resume in PDF, DOCX, or TXT format."
    )

    if resume_file:
        st.success(f"✅ Successfully uploaded `{resume_file.name}`")
        if st.button("Find Top Job Matches", type="primary", width='stretch'):
            with st.spinner(f"Sending your resume to the AI backend for matching..."):
                tmp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.name)[1]) as tmp_file:
                        tmp_file.write(resume_file.getvalue())
                        tmp_file_path = tmp_file.name
                    raw_resume_text = extract_text_from_file(tmp_file_path)

                    endpoint = f"{API_URL}/applicant/match/{model_choice.lower()}"
                    payload = {"raw_text": raw_resume_text, "top_k": top_k}
                    
                    response = requests.post(endpoint, json=payload, timeout=180) # 3-minute timeout
                    response.raise_for_status() # Raises HTTPError for bad responses eg. 4xx, 5xx

                    api_data = response.json()
                    matches = api_data.get("matches", [])
                    message = api_data.get("message", "No message from server.")

                    if not matches:
                        st.warning("⚠️ No suitable job matches found.")
                    else:
                        st.info(message)
                        st.subheader(f"Top {len(matches)} Job Matches:")
                        
                        df = pd.DataFrame(matches) # Pandas handles list of dicts perfectly
                        df = df.sort_values(by="match_score", ascending=False).reset_index(drop=True)

                        chart = alt.Chart(df).mark_bar().encode(
                            y=alt.Y('job_title', sort='-x', title=None, axis=alt.Axis(labelLimit=400)),
                            x=alt.X('match_score', axis=None, scale=alt.Scale(domainMin=0)),
                            tooltip=['job_title', alt.Tooltip('match_score', format='.3f')]
                        ).properties(title="Relative Job Match Scores").interactive()
                        
                        st.altair_chart(chart, use_container_width=True)

                except requests.exceptions.RequestException as e:
                    st.error(f"API Error:⚠️ Could not connect to the backend. Please ensure the backend server is running. Details: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

# Recruiter view of the app
if app_mode == "Recruiter":
    st.header("Recruiter: Rank Resumes for a Job Description")
    job_desc_file = st.file_uploader("Upload the job description", type=['pdf', 'docx', 'txt'])
    resume_files = st.file_uploader("Upload candidate resumes", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

    if job_desc_file and resume_files:
        st.success(f"✅ Successfully uploaded job description and {len(resume_files)} resumes.")
        if st.button("Rank Resumes", type="primary", width='stretch'):
            with st.spinner(f"Sending files to the AI backend for ranking..."):
                temp_dir = None
                job_desc_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(job_desc_file.name)[1]) as tmp_file:
                        tmp_file.write(job_desc_file.getvalue())
                        job_desc_path = tmp_file.name
                    raw_job_text = extract_text_from_file(job_desc_path)

                    temp_dir = tempfile.mkdtemp()
                    for resume_file in resume_files:
                        with open(os.path.join(temp_dir, resume_file.name), "wb") as f:
                            f.write(resume_file.getbuffer())
                    raw_resume_texts = bulk_load_raw_resume_files(temp_dir)

                    endpoint = f"{API_URL}/recruiter/rank/{model_choice.lower()}"
                    payload = {
                        "raw_job_text": raw_job_text,
                        "raw_resume_texts": raw_resume_texts,
                        "top_k": top_k
                    }
                    response = requests.post(endpoint, json=payload, timeout=300) # 5-minute timeout
                    response.raise_for_status() # Raises HTTPError for bad responses eg. 4xx, 5xx

                    api_data = response.json()
                    ranked_resumes = api_data.get("matches", [])
                    message = api_data.get("message", "No message from server.")

                    if not ranked_resumes:
                        st.warning("⚠️ Could not rank resumes. Please check the files.")
                    else:
                        st.info(message)
                        st.subheader(f"Top {len(ranked_resumes)} Ranked Resumes:")
                        df = pd.DataFrame(ranked_resumes)
                        df["match_score"] = df["match_score"].apply(lambda x: min(1.0, x))
                        st.dataframe(
                            df,
                            column_config={
                                "resume_filename": st.column_config.TextColumn("Resume"),
                                "match_score": st.column_config.ProgressColumn(
                                    "Match Score", format="%.2f", min_value=0, max_value=1
                                ),
                            },
                            hide_index=True,
                        )

                except requests.exceptions.RequestException as e:
                    st.error(f"API Error: Could not connect to the backend. Please ensure the backend server is running. Details: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if job_desc_path and os.path.exists(job_desc_path):
                        os.unlink(job_desc_path)
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
