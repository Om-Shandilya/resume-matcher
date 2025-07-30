from visualizing_data import (
    visualize_word_frequency,
    plot_wordcloud,
    plot_length_distribution,
    plot_similarity_heatmat,
    top_words_by_category)

def visualize_cleaned_data(df_resumes, df_jobs):
    print("⏳ Visualizing Resume Dataset...")
    visualize_word_frequency(df_resumes, number_of_words=30)
    plot_wordcloud(df_resumes)
    plot_length_distribution(df_resumes)
    plot_similarity_heatmat(df_resumes, number_of_samples=100)

    print("⏳ Visualizing Job Dataset...")
    visualize_word_frequency(df_jobs, number_of_words=30)
    plot_wordcloud(df_jobs)
    plot_length_distribution(df_jobs)

    top_words_by_category(
        df=df_jobs,
        text_column='text_cleaned',
        category_column='role',
        number_of_categories=10,
        number_of_words=10
    )