# 🎯 AI-Powered Resume-Job Matcher

Welcome to the official repository for the AI-Powered Resume-Job Matcher! This project is a comprehensive, full-stack application designed to streamline the recruitment process by intelligently matching candidates to job opportunities. After a long journey of development, debugging, and deployment, we're thrilled to present a robust and scalable solution.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Frameworks](https://img.shields.io/badge/Frameworks-FastAPI%20%7C%20Streamlit-green.svg)](https://fastapi.tiangolo.com/)

---

## 🚀 Live Demos

Experience the application live! The project is deployed as two separate, communicating services on Hugging Face Spaces.

* **Frontend GUI (Streamlit App)**: [**Access the Live Application Here**](https://huggingface.co/spaces/Om-Shandilya/resume-matcher-app)
* **Backend API (FastAPI Docs)**: [**Explore the API Documentation Here**](https://huggingface.co/spaces/Om-Shandilya/resume-matcher-api/docs)

---

## ✨ Features

This project offers a dual-interface system catering to both job applicants and recruiters.

### For the Applicant:
* **Find Your Fit**: Upload your resume in `.pdf`, `.docx`, or `.txt` format.
* **Get Instant Matches**: The system analyzes your resume and matches it against a database of 147 unique job titles.
* **Dual Model Power**: Choose between a fast, baseline **TF-IDF** model or a more accurate, semantically-aware **BERT** model (`all-MiniLM-L6-v2`) for your results.

### For the Recruiter:
* **Efficient Candidate Sourcing**: Upload a single job description and a batch of multiple candidate resumes.
* **Automated Ranking**: The system ranks all submitted resumes based on their relevance to the job description, saving hours of manual review.
* **Flexible Analysis**: Leverage either the TF-IDF or the advanced BERT model to score and rank your talent pool.

---

## 🏛️ Architecture

This application is built using a modern, decoupled, client-server architecture, ensuring scalability and maintainability.



* **Frontend**: A user-friendly and interactive web interface built with **Streamlit**. It is responsible for all user interactions and visualizations.
* **Backend**: A high-performance, robust RESTful API built with **FastAPI**. It handles all the heavy lifting, including text processing, model inference, and the core matching logic.
* **Communication**: The frontend communicates with the backend via standard HTTP requests, sending data as JSON payloads.
* **Model Hosting**: All machine learning artifacts (models, vectorizers, and data) are hosted on the **Hugging Face Hub**, keeping the application code lightweight and portable.

---

## 💻 Tech Stack

* **Backend**: Python, FastAPI, Uvicorn
* **Frontend**: Streamlit, Altair
* **ML/NLP**: Scikit-learn, Sentence-Transformers, PyTorch, FAISS, NLTK
* **Data Handling**: Pandas, Joblib
* **Deployment**: Docker, Hugging Face Spaces, Git LFS

---

## 🛠️ Local Setup and Installation

To run this project on your local machine, please follow these steps.

### 1. Prerequisites
* Conda installed on your system.
* Git and Git LFS installed.

### 2. Clone the Repository
```bash
git clone [https://github.com/Om-Shandilya/resume-matcher](https://github.com/Om-Shandilya/resume-matcher)
cd resume-matcher