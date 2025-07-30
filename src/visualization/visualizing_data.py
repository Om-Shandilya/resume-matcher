import os
import re
from datetime import datetime
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

def handle_plot(save_path: str = '../data/saved_plots', 
                filename: str = None, 
                label: str = None,
                add_timestamp: bool = True,
                show_plot: bool = True):
    """
    Handles saving or displaying a plot based on provided parameters, with optional timestamp in filename.

    Args:
        save_path (str): Directory to save the plot. 
        filename (str): Name of the file to save the plot (e.g., 'plot.png'). If None, the plot will be displayed.
        label (str): [Required with {filename} to save] Label to prepend to the filename (e.g., 'jobs', 'resumes').
                     If None, the plot will be displayed instead.
        add_timestamp (bool): Whether to add a timestamp to the filename. Default is True.

    Returns:
        None: Displays the plot or saves it to the specified path.
    """
    if label and filename:
        os.makedirs(save_path, exist_ok=True)
        base, ext = os.path.splitext(filename)

        if not ext:
            ext = ".png"

        # Add label if given
        label_prefix = f"{label}_" if label else ""

        # Add timestamp if requested
        timestamp_suffix = f"_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}" if add_timestamp else ""

        # Final filename
        final_name = f"{label_prefix}{base}{timestamp_suffix}{ext}"
        save_full_path = os.path.join(save_path, final_name)

        plt.tight_layout()
        plt.savefig(save_full_path, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {save_full_path}")

        if show_plot:
            plt.show()
            
    else:
        plt.tight_layout()
        plt.show()

def visualize_word_frequency(
        df: pd.DataFrame,
        column_to_visualize: str ='text_cleaned',
        number_of_words: int = 30,
        save_path: str = '../data/saved_plots/word_frequencies',
        plot_label: str = None):
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
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.title(f"Top {number_of_words} Words in Resume Corpus")
    handle_plot(save_path, filename = f'word_freq_top_{number_of_words}.png', label=plot_label)


def plot_wordcloud(
        df: pd.DataFrame,
        column_to_visualize: str ='text_cleaned',
        save_path: str = '../data/saved_plots/wordclouds',
        plot_label: str = None,):
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
    plt.title("WordCloud")
    handle_plot(save_path, f'wordcloud.png', label=plot_label)


def plot_length_distribution(
        df: pd.DataFrame,
        column_to_visualize: str ='text_cleaned',
        number_of_bins: int = 30,
        save_path: str = '../data/saved_plots/length_distributions',
        plot_label: str = None):
    """
    Visualizes the distribution of text lengths in a specified column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_to_visualize (str): Column name to visualize. Default is 'text_cleaned'.

    Returns:
        None: Displays a histogram of text lengths.
    """

    # Create a new column with the length of each text
    text_lens = df[column_to_visualize].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)


    # Plot length distributions
    plt.figure(figsize=(10, 4))
    plt.hist(text_lens, bins=number_of_bins, alpha=0.7)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.title("Document Length Distribution")
    handle_plot(save_path, f'length_distribution.png', label=plot_label)


def plot_similarity_heatmat(df: pd.DataFrame,
                            column_to_visualize: str ='text_cleaned',
                            number_of_samples: int = 100,
                            max_features: int = 100,
                            save_path: str = '../data/saved_plots/similarity_heatmaps',
                            plot_label: str = None):
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
    handle_plot(save_path, f'similarity_heatmap.png', label=plot_label)

def top_words_by_category(df: pd.DataFrame,
                          text_column: str,
                          category_column: str,
                          number_of_categories: int = 5,
                          number_of_words: int = 10,
                          save_path: str = '../data/saved_plots/top_words_by_categories',
                          plot_label: str = None):
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
        plt.title(f"Top {number_of_words} Words in '{cat}' ({plot_label.capitalize()})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        safe_cat = re.sub(r'[^\w\-_.]', '_', cat)
        handle_plot(save_path, filename = f"top_{number_of_words}_words_in_{safe_cat}.png", label=plot_label)
