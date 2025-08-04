import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import Optional, Tuple


def get_tfidf_vectorizer(max_features: int = 5000, 
                         ngram_range: Tuple[int, int] = (1, 2)):
    """
    Creates a TF-IDF vectorizer with specified parameters.
    """
    return TfidfVectorizer(
           stop_words='english',
           lowercase=True,
           max_features=7000,        # Slightly larger vocabulary
           ngram_range=(1, 2),       # Add bigrams to capture phrases
           min_df=3,                 # Remove ultra-rare tokens
           max_df=0.85,              # Remove very common tokens
           norm='l2',                # Normalize for cosine similarity
    )

def save_vectorizer(vectorizer: TfidfVectorizer, 
                    path: str = '../models/tfidf_vectorizer.pkl'):
    
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


def vectorize_text(df: pd.DataFrame, 
                   text_column: str, 
                   label: str,  # e.g., 'resumes' or 'jobs'
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


def load_vectorizer(path: str):
    """
    Loads a vectorizer from a .pkl file.
    """
    if not path.endswith('.pkl'):
        path += '.pkl'
    return joblib.load(path)


def load_vector_data(path: str):
    """
    Loads a sparse matrix from a .npz file.
    """
    if not path.endswith('.npz'):
        path += '.npz'
    return load_npz(path)