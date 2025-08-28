import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download


def get_tfidf_vectorizer(max_features: int = 5000, 
                         ngram_range: Tuple[int, int] = (1, 2)):
    """
    Creates a TF-IDF vectorizer with specified parameters.
    """
    return TfidfVectorizer(
           stop_words='english',
           lowercase=True,
           max_features=max_features,        # Slightly larger vocabulary
           ngram_range=ngram_range,          # Add bigrams to capture phrases
           min_df=3,                         # Remove ultra-rare tokens
           max_df=0.85,                      # Remove very common tokens
           norm='l2',                        # Normalize for cosine similarity
    )

def get_combined_tfidf_vectorizer(max_features: int = 40000, 
                         ngram_range: Tuple[int, int] = (1, 2)):
    """
    Creates a TF-IDF vectorizer with specified parameters for larger vocab with both Jobs and Resume.
    """
    return TfidfVectorizer(
           stop_words="english",            # Remove common English stopwords
           lowercase=True,                  # Convert all to lowercase
           max_features=max_features,       # Balanced for resumes + jobs
           ngram_range=ngram_range,         # By default Unigrams + Bigrams
           min_df=5,                        # Ignore very rare words   
           max_df=0.85,                     # Ignore very common words
           sublinear_tf=True,               # Smooth term frequency scaling
           norm="l2"                        # Normalize for cosine similarity
)

def save_vectorizer(vectorizer: TfidfVectorizer, 
                    path: str = 'models/tfidf/dev_tfidf/tfidf_vectorizer.pkl'):
    
    """
    Saves a TfidfVectorizer object to a given path. Appends .pkl if missing.
    """
    # Ensure .pkl extension
    if not path.endswith('.pkl'):
        path += '.pkl'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"✅ TF-IDF vectorizer saved to: [{path}]")


def save_vector_data(matrix: csr_matrix, path: str):
    """
    Saves a sparse TF-IDF matrix to a .npz file.
    """
    if not path.endswith('.npz'):
        path += '.npz'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_npz(path, matrix)
    print(f"✅ TF-IDF matrix saved to: [{path}]")


def tfidf_vectorize_text(df: pd.DataFrame, 
                   text_column: str, 
                   label: str,
                   vectorizer: Optional[TfidfVectorizer] = None,
                   fit_vectorizer: bool = False, 
                   save_path: Optional[str] = None,
                   save_vectorizer_file: bool = False):
    """
    Transforms a DataFrame's text column into TF-IDF features and saves them if a path is provided.

    To save the vectorizer and matrix, ensure 'save_path' is provided along with a valid 'label'.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column to vectorize.
        label (str): Label prefix for saved files (e.g., 'resumes', 'jobs').
        vectorizer (Optional[TfidfVectorizer]): TF-IDF vectorizer to use (optional).
        fit_vectorizer (bool): Fit or just transform.
        save_path (Optional[str]): Directory to save vectorizer/matrix files.

    Returns:
        tuple: (sparse_matrix, fitted_vectorizer)
    """

    if df[text_column].isnull().any():
        print(f"\n⚠️ Found missing values in column '{text_column}', replacing with empty string.")
        df[text_column] = df[text_column].fillna("")

    if vectorizer is None:
        vectorizer = get_tfidf_vectorizer()

    if fit_vectorizer:
        X = vectorizer.fit_transform(df[text_column])
    else:
        X = vectorizer.transform(df[text_column])

    if save_path and label:
        save_vector_data(X, os.path.join(save_path, f"{label}_tfidf_matrix.npz"))
        if save_vectorizer_file:
            save_vectorizer(vectorizer, os.path.join(save_path, f"{label}_tfidf_vectorizer.pkl"))

    return X, vectorizer


def load_tfidf_vectorizer(local_vectorizer_path: str, repo_id: str, filename: str):
    """Load TF-IDF vectorizer, preferring local then HF Hub."""
    if local_vectorizer_path:
        if not os.path.exists(local_vectorizer_path):
            raise FileNotFoundError(f"❌ Local TF-IDF vectorizer not found at {local_vectorizer_path}")
        print(f"📂 Loading local TF-IDF vectorizer from {local_vectorizer_path}")
        return joblib.load(local_vectorizer_path)

    print(f"🌐 Downloading TF-IDF vectorizer from Hugging Face Hub ({repo_id}/{filename})")
    vec_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return joblib.load(vec_path)

def load_tfidf_matrix(local_matrix_path: str, repo_id: str, filename: str):
    """Load TF-IDF matrix, preferring local then HF Hub."""
    if local_matrix_path:
        if not os.path.exists(local_matrix_path):
            raise FileNotFoundError(f"❌ Local TF-IDF matrix not found at {local_matrix_path}")
        print(f"📂 Loading local TF-IDF matrix from {local_matrix_path}")
        return load_npz(local_matrix_path)

    print(f"🌐 Downloading TF-IDF matrix from Hugging Face Hub ({repo_id}/{filename})")
    mat_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_npz(mat_path)