import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

def visualize_word_frequency(
        df: pd.DataFrame,
        column_to_visualize: str ='text_cleaned',
        number_of_words: int = 30):
    """
    Visualizes word frequency in a specified column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_to_visualize (str): Column name to visualize. Default is 'text_cleaned'.
        top_words (int): Number of top words to display. Default is 30.
    
    Returns:
        None: Displays a bar plot of the top specified number of words.
    """
    # extract words from the specified column as str
    words = df[column_to_visualize].dropna().tolist()

    # Combine all cleaned text into one
    all_words = " ".join(words)

    # Get frequency
    word_freq = Counter(all_words.split())

    # Top specified number of words
    # Top specified number of words
    top_words = {word: freq for word, freq in word_freq.most_common(number_of_words) if word.strip() != ""}


    # Bar Plot
    plt.figure(figsize=(12, 6))
    plt.bar(top_words.keys(), top_words.values())
    plt.xlabel("words")
    plt.ylabel("frequency")
    plt.xticks(rotation=45)
    plt.title(f"Top {number_of_words} Words in Resume Corpus")
    plt.show()


def plot_wordcloud(
        df: pd.DataFrame,
        column_to_visualize: str ='text_cleaned'):
    """
    Visualizes a word cloud from a specified column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_to_visualize (str): Column name to visualize. Default is 'text_cleaned'.

    Returns:
        None: Displays a word cloud of the words in the specified column.
    """

    # extract words from the specified column as str
    words = df[column_to_visualize].dropna().tolist()

    # Combine all cleaned text into one
    all_words = " ".join(words)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Resume WordCloud")
    plt.show()


def plot_length_distribution(
        df: pd.DataFrame,
        column_to_visualize: str ='text_cleaned',
        number_of_bins: int = 30):
    """
    Visualizes the distribution of text lengths in a specified column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_to_visualize (str): Column name to visualize. Default is 'text_cleaned'.

    Returns:
        None: Displays a histogram of text lengths.
    """

    # Create a new column with the length of each text
    df['text_len'] = df[column_to_visualize].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)


    # Plot length distributions
    plt.figure(figsize=(10, 4))
    plt.hist(df['text_len'], bins=number_of_bins, alpha=0.7)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.title("Document Length Distribution")
    plt.show()


def plot_similarity_heatmat(df: pd.DataFrame,
                          column_to_visualize: str ='text_cleaned',
                          number_of_samples: int = 100,
                          max_features: int = 100):
    """
    Visualizes the cosine similarity heatmap of documents in a specified column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_to_visualize (str): Column name to visualize. Default is 'text_cleaned'.
        number_of_samples (int): Number of samples to visualize. Default is 100.
        max_features (int): Maximum number of features for TF-IDF vectorization. Default is 100.
    
    Returns:
        None: Displays a heatmap of cosine similarity between documents.
    """

    vec = TfidfVectorizer(max_features=max_features)

    # Extract words from the specified column and fit and trasform as TF-IDF matrix (or a vector)
    texts = df[column_to_visualize].dropna().iloc[:number_of_samples].tolist()
    tfidf_matrix = vec.fit_transform(texts)

    # Calculate and plot cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    sns.heatmap(similarity_matrix, cmap='coolwarm')
    plt.title("Resume-to-Resume Similarity Heatmap")
    plt.show()

def top_words_by_category(df: pd.DataFrame,
                          text_column: str,
                          category_column: str,
                          number_of_categories: int = 10,
                          number_of_words: int = 10):
    """
    Visualizes the top words in each of the most frequent categories of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column name containing the text data.
        category_column (str): Column name containing the category data.
        number_of_categories (int): Number of most frequent categories to visualize. Default is 10.
        number_of_words (int): Number of top words to display for each category. Default is 10.

    Returns:
        None: Displays a bar plot of the top words in each category.
    """

    # Step 1: Identify top-N most frequent categories
    top_categories = df[category_column].value_counts().head(number_of_categories).index.tolist()

    # Step 2: Loop through these top categories only
    for cat in top_categories:
        subset = df[df[category_column] == cat]

        # Vectorize the text
        vec = CountVectorizer(stop_words='english', max_features=1000)
        X = vec.fit_transform(subset[text_column])

        # Get word frequencies
        word_freq = X.sum(axis=0).A1
        words = vec.get_feature_names_out()
        freq_dict = dict(zip(words, word_freq))

        # Select top-N words
        top_n = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:number_of_words])

        # Plot
        plt.figure(figsize=(10, 4))
        plt.bar(top_n.keys(), top_n.values(), color='teal')
        plt.title(f"Top {number_of_words} Words in '{cat}'")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
