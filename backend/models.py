from pydantic import BaseModel
from typing import List, Dict


# Applicant Side Models:
class ResumeRequest(BaseModel):
    """The request body for matching a single resume."""
    raw_text: str
    top_k: int | None = None

class JobMatch(BaseModel):
    """Represents a single job match with its score."""
    job_title: str
    match_score: float

class ApplicantResponse(BaseModel):
    """The response body containing job matches and a message."""
    matches: List[JobMatch]
    message: str


# Recruiter Side Models:
class RecruiterRequest(BaseModel):
    """The request body for ranking multiple resumes against a job description."""
    raw_job_text: str
    raw_resume_texts: Dict[str, str] # dict of {filename: raw_resume_text}
    top_k: int | None = None

class ResumeMatch(BaseModel):
    """Represents a single ranked resume with its score."""
    resume_filename: str
    match_score: float

class RecruiterResponse(BaseModel):
    """The response body containing ranked resumes and a message."""
    matches: List[ResumeMatch]
    message: str