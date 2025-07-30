from visualizing_data import (
    visualize_word_frequency,
    plot_wordcloud,
    plot_length_distribution,
    plot_similarity_heatmat,
    top_words_by_category)
import pandas as pd


def visualize_cleaned_data(df_resumes, df_jobs, save_plots=True):
    print("⏳ Visualizing Resume Dataset...")
    label = "resume" if save_plots else None
    visualize_word_frequency(df_resumes, number_of_words=30, plot_label=label)
    plot_wordcloud(df_resumes, plot_label=label)
    plot_length_distribution(df_resumes, plot_label=label)
    plot_similarity_heatmat(df_resumes, number_of_samples=100, plot_label=label)
    print("✅ Resume Dataset Visualization Complete!")

    label = "job" if save_plots else None
    print("⏳ Visualizing Job Dataset...")
    visualize_word_frequency(df_jobs, number_of_words=30, plot_label=label)
    plot_wordcloud(df_jobs, plot_label=label)
    plot_length_distribution(df_jobs, plot_label=label)

    top_words_by_category(df=df_jobs,
        text_column='text_cleaned',
        category_column='role',
        number_of_categories=5,
        number_of_words=10,
        plot_label=label
    )
    print("✅ Job Dataset Visualization Complete!")

if __name__ == "__main__":
    from src.data.loading_data import load_or_clean_resume_data, load_or_clean_job_data

    df_resumes = load_or_clean_resume_data()
    df_jobs = load_or_clean_job_data(sample_size=2484)

    visualize_cleaned_data(df_resumes, df_jobs, save_plots=True)