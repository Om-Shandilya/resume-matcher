import re
import string
import pandas as pd
from typing import Optional
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def get_wordnet_pos(tag):
    """
    Map NLTK POS tags to WordNet POS tags for better lemmatization.

    args:
        tag (str): NLTK POS tag which is a Treebank tag so we just look at the first letter 
                   to get the simpler WordNet POS tag.
    """

    if tag.startswith('J'):                 # Recognize adjectives
        return wordnet.ADJ
    elif tag.startswith('V'):               # Recognize verbs
        return wordnet.VERB
    elif tag.startswith('N'):               # Recognize nouns
        return wordnet.NOUN
    elif tag.startswith('R'):               # Recognize adverbs
        return wordnet.ADV
    else:
        return wordnet.NOUN                 # Else set default to noun

def clean_text(
    text: str,
    remove_numbers: bool = True,
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True):
    """
    Clean input text with configurable options.

    Args:
        text (str): The input text to clean.
        remove_stopwords (bool): Whether to remove stopwords.
        apply_lemmatization (bool): Whether to apply lemmatization.
        remove_numbers (bool): Whether to remove/keep number.

    Returns:
        str: The cleaned text.
    """

    if pd.isnull(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Optionally remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Tokenize by whitespace
    tokens = text.split()

    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    # Apply lemmatization if requested
    if apply_lemmatization:
        # Tag all tokens at once
        tagged_tokens = pos_tag(tokens)

        # Cache POS-tag to lemmatized word
        lemmatized_words = []
        cache = {}

        for word, tag in tagged_tokens:
            key = (word, tag)
            if key not in cache:
                wn_tag = get_wordnet_pos(tag)
                cache[key] = lemmatizer.lemmatize(word, wn_tag)
            lemmatized_words.append(cache[key])

        tokens = lemmatized_words

    return ' '.join(tokens)

def clean_column(
    df: pd.DataFrame,
    column_name: str,
    new_column_name: Optional[str] = None,
    remove_numbers: bool = True,
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True):
    """
    Applies clean_text to an entire column of a DataFrame.

    Parameters:
        df: pandas DataFrame.
        column_name: str, the column to clean.
        remove_stopwords: bool, whether to remove stopwords.
        apply_lemmatization: bool, whether to apply lemmatization.
        remove_numbers: bool, whether to keep /remove numbers.
        new_column_name: str or None, where to save cleaned column. If None, overwrite original.

    Returns:
        DataFrame with updated or new column.
    """

    # Check if column name is specified or not and act accordingly
    if new_column_name is None:
        new_column_name = column_name + '_cleaned'

    # Apply clean_text() to all entries in the specified column
    df[new_column_name] = df[column_name].apply(
        lambda x: clean_text(
            x,
            remove_numbers = remove_numbers,
            remove_stopwords = remove_stopwords,
            apply_lemmatization = apply_lemmatization
        )
    )
    return df