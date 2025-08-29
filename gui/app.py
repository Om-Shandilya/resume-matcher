import streamlit as st
import os
import tempfile
import pandas as pd
import shutil
import sys
import altair as alt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.utils.bulk_loading import bulk_load_raw_resume_files
from src.utils.file_reader import extract_text_from_file
from src.utils.model_loader import get_applicant_matrix, get_applicant_vectorizer, get_bert_model, get_faiss_index, get_recruiter_vectorizer
from pipelines.core.applicant import run_tfidf_pipeline as applicant_tfidf, run_bert_pipeline as applicant_bert
from pipelines.core.recruiter import rank_with_tfidf as recruiter_tfidf, rank_with_bert as recruiter_bert

# --- App Configuration ---
st.set_page_config(
    page_title="Resume-Job Matcher",
    page_icon="📄",
    layout="wide"
)

# --- Main App ---
st.title("🎯 AI-Powered Resume-Job Matcher")
st.write("---")

# --- Sidebar for Mode Selection ---
with st.sidebar:
    st.header("Controls")
    app_mode = st.radio(
        "Choose your view",
        ("Applicant", "Recruiter"),
        help="Select 'Applicant' to match your resume to jobs. Select 'Recruiter' to rank resumes for a job."
    )
    model_choice = st.selectbox(
        "Choose the AI Model",
        ("TF-IDF", "BERT"),
        help="TF-IDF is faster. BERT is more accurate."
    )

    st.write("---")

    # Add a checkbox to control the 'show all' feature
    show_all = st.checkbox("Show all matches", value=False)

    if show_all:
        top_k = None
        # Disable the slider when 'show_all' is checked for better UX
        st.slider(
            "Number of matches to show", 
            min_value=1, max_value=50, value=5, step=1,
            disabled=True
        )
        st.info("Showing all ranked results.")
    else:
        # Enable the slider when 'show_all' is unchecked
        top_k = st.slider(
            "Number of matches to show", 
            min_value=1, max_value=50, value=5, step=1,
            disabled=False
        )


# --- Applicant View ---
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
            
            with st.spinner(f"Analyzing resume with {model_choice}..."):
                
                tmp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.name)[1]) as tmp_file:
                        tmp_file.write(resume_file.getvalue())
                        tmp_file_path = tmp_file.name

                    raw_resume_text = extract_text_from_file(tmp_file_path)

                    if model_choice == "BERT":
                        bert_model = get_bert_model()
                        faiss_index = get_faiss_index()
                        matches, message = applicant_bert(raw_resume_text,
                                                          model=bert_model,
                                                          job_index=faiss_index,
                                                          top_k=top_k,)
                        
                    else:
                        applicant_vectorizer = get_applicant_vectorizer()
                        applicant_matrix = get_applicant_matrix()
                        matches, message = applicant_tfidf(raw_resume_text,
                                                           vectorizer=applicant_vectorizer,
                                                           job_matrix=applicant_matrix,
                                                           top_k=top_k)

                    if not matches:
                        st.warning("⚠️ No suitable job matches found.")
                    else:
                        st.subheader(f"Top {len(matches)} Job Matches:")
                        st.info(message)

                        df = pd.DataFrame(matches, columns=["Job Title", "Match Score"])

                        # Sort the DataFrame by 'Match Score' in descending order to show best matches at the top
                        df = df.sort_values(by="Match Score", ascending=False).reset_index(drop=True)

                        chart = alt.Chart(df).mark_bar().encode(
                            y=alt.Y('Job Title', sort='-x', title=None),
                            x=alt.X('Match Score', axis=None, scale=alt.Scale(domainMin=0)), 
                            
                            # Tooltip to reveal score on hover
                            tooltip=['Job Title', alt.Tooltip('Match Score', format='.3f')]
                        ).properties(
                            # Set a responsive title for the chart to indicate what the bars represent
                            title="Relative Job Match Scores" 
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)


# --- Recruiter View ---
if app_mode == "Recruiter":
    st.header("Recruiter: Rank Resumes for a Job Description")
    
    job_desc_file = st.file_uploader(
        "Upload the job description",
        type=['pdf', 'docx', 'txt'],
        help="Upload the job description in PDF, DOCX, or TXT format."
    )
    
    resume_files = st.file_uploader(
        "Upload candidate resumes",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload one or more resumes."
    )

    if job_desc_file and resume_files:
        st.success(f"✅ Successfully uploaded job description `{job_desc_file.name}` and {len(resume_files)} resumes.")
        if st.button("Rank Resumes", type="primary", width='stretch'):
            
            with st.spinner(f"Ranking {len(resume_files)} resumes with {model_choice}..."):
                
                # Paths for cleanup in the finally block
                temp_dir = None
                job_desc_path = None
                
                try:
                    # 1. Handle the single job description file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(job_desc_file.name)[1]) as tmp_file:
                        tmp_file.write(job_desc_file.getvalue())
                        job_desc_path = tmp_file.name
                    raw_job_text = extract_text_from_file(job_desc_path)

                    # 2. Handle multiple resume files by creating a temp directory for bulk loading
                    temp_dir = tempfile.mkdtemp()
                    for resume_file in resume_files:
                        resume_path = os.path.join(temp_dir, resume_file.name)
                        with open(resume_path, "wb") as f:
                            f.write(resume_file.getbuffer())
                    
                    # Bulk loading all resumes from the temp directory
                    raw_resume_texts = bulk_load_raw_resume_files(temp_dir)

                    # 3. Call the appropriate model's pipeline based on the model choice (default to TF-IDF)
                    if model_choice == "BERT":
                        bert_model = get_bert_model()
                        ranked_resumes, message = recruiter_bert(raw_job_text,
                                                                 raw_resume_texts,
                                                                 model=bert_model,
                                                                 top_k=top_k)
                    else:
                        vectorizer = get_recruiter_vectorizer()
                        ranked_resumes, message = recruiter_tfidf(raw_job_text,
                                                                  raw_resume_texts,
                                                                  vectorizer=vectorizer,
                                                                  top_k=top_k)

                    # 4. Display results
                    if not ranked_resumes:
                        st.warning("⚠️ Could not rank resumes. Please check the files.")
                    else:
                        st.subheader(f"Top {len(ranked_resumes)} Ranked Resumes:")
                        st.info(message)
                        df = pd.DataFrame(ranked_resumes, columns=["Resume", "Match Score"])
                        
                        df["Match Score"] = df["Match Score"].apply(lambda x: min(1.0, x))
                        st.dataframe(
                            df,
                            column_config={"Resume": st.column_config.TextColumn("Resume"),
                                           "Match Score": st.column_config.ProgressColumn("Match Score",
                                                format="%.2f",
                                                min_value=0,
                                                max_value=1,),
                                           },
                                            width='stretch',
                                            hide_index=True,
                        )

                except Exception as e:
                    st.error(f"⚠️An error occurred: {e}")
                
                finally:
                    # 5. Clean up all temporary files and the directory
                    if job_desc_path and os.path.exists(job_desc_path):
                        os.unlink(job_desc_path)
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)