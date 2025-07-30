import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
