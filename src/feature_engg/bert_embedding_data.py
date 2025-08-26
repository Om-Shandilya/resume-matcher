import os
import numpy as np
import pandas as pd
import torch
import faiss
import torch
from faiss import read_index
from typing import Optional
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download


def get_bert_model(model_name: str = "all-MiniLM-L6-v2",
                   device: str = None):
    """
    Loads a BERT-based sentence transformer model for embeddings.

    Args:
        model_name (str): HuggingFace model name. Default is "all-MiniLM-L6-v2".
        device (str, optional): "cuda", "cpu", or None (auto-detect).

    Returns:
        SentenceTransformer: Loaded model ready for encoding.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(model_name, device=device)


def save_bert_embeddings(embeddings: np.ndarray,
                         path: str):
    """
    Save dense BERT embeddings as a FAISS index file (.faiss).
    """

    if not path.endswith('.faiss'):
        path += '.faiss'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)  # Inner Product (cosine if normalized)
    index.add(embeddings)

    faiss.write_index(index, path)
    print(f"✅ BERT embeddings saved to FAISS index: [{path}] "
          f"with {index.ntotal} vectors, dim={embedding_dimension}")


def save_bert_model(vectorizer: SentenceTransformer,
                    path: str):
    """Save the full SentenceTransformer model to disk."""

    os.makedirs(path, exist_ok=True)
    vectorizer.save(path)
    print(f"✅ BERT model saved to: [{path}]")


def bert_embed_text(df: pd.DataFrame,
                   text_column: str,
                   label: str = 'jobs',  # default is 'jobs' as most common use-case in pipeline.
                   model: Optional[SentenceTransformer] = None,
                   save_path: Optional[str] = None,
                   save_model_file: bool = False):
    """
    Encodes text from a DataFrame into dense BERT embeddings.

    To save the embeddings and model, ensure 'save_path' is provided along with a valid 'label'.

    Args:
        df (pd.DataFrame): DataFrame containing the text to be encoded.
        text_column (str): Column with text to be encoded.
        label (str): Label prefix for saved files (e.g., 'resumes', 'jobs').
        model (SentenceTransformer, optional): Preloaded model.
        save_path (str, optional): Directory to save outputs.
        save_model_file (bool): If True, also saves the model reference.

    Returns:
        tuple: (embeddings ndarray, model) 
    """

    if df[text_column].isnull().any():
        print(f"\n⚠️ Found missing values in column '{text_column}', replacing with empty string.")
        df[text_column] = df[text_column].fillna("")

    if model is None:
        model = get_bert_model()

    embeddings = model.encode(
        df[text_column].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True   # normalizing as it is good for cosine similarity.
    )

    if save_path and label:
        save_bert_embeddings(embeddings, os.path.join(save_path, f"{label}.faiss"))
        if save_model_file:
            save_bert_model(model, save_path)

    return embeddings, model


def load_faiss_index(local_index_path: str, repo_id: str, filename: str):
    """Load FAISS index, preferring local then HF Hub."""
    if local_index_path and os.path.exists(local_index_path):
        print(f"📂 Loading local FAISS index from {local_index_path}")
        return read_index(local_index_path)
    else:
        print(f"🌐 Downloading FAISS index from Hugging Face Hub ({repo_id})")
        faiss_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return read_index(faiss_path)

def load_bert_model(local_model_path: str, repo_id: str):
    """Load BERT model, preferring local then HF Hub."""
    if local_model_path and os.path.exists(local_model_path):
        print(f"📂 Loading local BERT model from {local_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModel.from_pretrained(local_model_path)
    else:
        print(f"🌐 Downloading BERT model from Hugging Face Hub ({repo_id})")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModel.from_pretrained(repo_id)
    return tokenizer, model

def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]  # First element is [batch, seq_len, hidden_dim]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def convert_hf_model_to_st(hf_model_path: str,
                           st_model_path: str):
    """
    Converts a HuggingFace model to a SentenceTransformer model.

    Needed as fine-tuning was performed using HuggingFace's Transformers library.

    Args:
        hf_model_path (str): Path to the HuggingFace model.
        st_model_path (str): Path to save the SentenceTransformer model.
    
    Returns:
        None: Saves the SentenceTransformer model to the specified path.
    """
    # Build SentenceTransformer from HF model
    word_embedding_model = models.Transformer(hf_model_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Save to the provided path
    os.makedirs(st_model_path, exist_ok=True)
    st_model.save(st_model_path)
    print(f"✅ Converted HuggingFace model [{hf_model_path}] "
          f"to SentenceTransformer at [{st_model_path}]")