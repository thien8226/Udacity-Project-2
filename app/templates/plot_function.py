import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
import plotly.graph_objs as go
from plotly.graph_objs import Bar
import plotly.express as px

# Ensure necessary NLTK components are downloaded
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])

def genres_distribution_plot(df):
    """
    Generates a bar plot for the distribution of message genres within a DataFrame.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the 'genre' and 'message' columns.

    Returns:
    dict: A dictionary containing the plotly graph objects for the plot.
    """
    genre_counts = df.groupby('genre').count()['message']  # Count occurrences of each genre
    genre_names = list(genre_counts.index)  # Extract genre names

    # Plot configuration using plotly
    infor_dict = {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {'title': 'Distribution of Message Genres', 'yaxis': {'title': "Count"}, 'xaxis': {'title': "Genre"}}
        }
    fig = go.Figure(data=infor_dict['data'], layout=infor_dict['layout'])
    
    return fig

def label_distribution_plot(df):
    """
    Generates a bar plot for the distribution of various labels within a DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame where each column represents a label and contains numeric values.

    Returns:
    dict: A dictionary containing the plotly graph objects for the plot.
    """
    # List of labels in the DataFrame
    label_names=['other_infrastructure', 'food', 'security', 'shelter', 'refugees', 'earthquake', 
              'medical_products', 'missing_people', 'death', 'shops', 'related', 'offer', 
              'medical_help', 'hospitals', 'aid_centers', 'weather_related', 'cold', 'request', 
              'storm', 'water', 'floods', 'military', 'other_aid', 'aid_related', 'direct_report', 
              'fire', 'clothing', 'money', 'buildings', 'transport', 'tools', 'search_and_rescue', 
              'other_weather', 'child_alone', 'infrastructure_related', 'electricity']

    label_counts = [df[x].sum() for x in label_names]  # Sum each label's occurrences
    label_counts_series = pd.Series(label_counts, index=label_names)

    # Plot configuration using plotly
    infor_dict = {
            'data': [Bar(x=label_names, y=label_counts_series)],
            'layout': {'title': 'Distribution of Label', 'yaxis': {'title': "Count"}, 'xaxis': {'title': "Label"}}
        }
    fig = go.Figure(data=infor_dict['data'], layout=infor_dict['layout'])
    
    return fig

def tokenize(text):
    """
    Tokenizes and lemmatizes the input text using NLTK's word_tokenize and WordNetLemmatizer.

    Args:
    text (str): Text to be tokenized and lemmatized.

    Returns:
    list: A list of cleaned tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

def word_cloud_plot(df):
    """
    Creates a word cloud visualization using TF-IDF scores of words from messages in a DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing a column 'message' with text data.

    Returns:
    tuple: A plotly figure object for the word cloud and the DataFrame containing TF-IDF features.
    """
    # Text processing
    CV = CountVectorizer(tokenizer=tokenize)
    TT = TfidfTransformer()
    text_data = df['message']
    count_matrix = CV.fit_transform(text_data)
    tfidf_matrix = TT.fit_transform(count_matrix)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=CV.get_feature_names_out())

    # Summing up the TF-IDF scores for each word and filtering
    word_count_dict = tfidf_df.sum(axis=0).to_dict()
    unwanted_chars = set('!@#$%^&*()-=+[]{}\\|;:"\'<>,.?/')
    filtered_vocabulary = {key: value for key, value in word_count_dict.items() if key not in unwanted_chars}
    top_10 = sorted(filtered_vocabulary.items(), key=lambda item: item[1], reverse=True)[:50]

    # Normalize TF-IDF scores for plotting
    word_scores = dict(top_10)
    max_font_size = 50
    min_font_size = 10
    max_score = max(word_scores.values())
    min_score = min(word_scores.values())
    normalize = lambda score: min_font_size + (score - min_score) / (max_score - min_score) * (max_font_size - min_font_size)
    color_scale = px.colors.sequential.Blues  # Color scale
    normalized_indices = np.linspace(0, len(color_scale) - 1, len(word_scores), dtype=int)

    # Creating the plot
    fig = go.Figure()
    for idx, (word, score) in enumerate(word_scores.items()):
        color_idx = normalized_indices[len(normalized_indices)-1-idx]
        fig.add_trace(go.Scatter(x=[np.random.rand()], y=[np.random.rand()], text=word,
            mode='text', textfont=dict(size=normalize(score), color=color_scale[color_idx]), showlegend=False))

    fig.update_layout(xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
                      yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
                      title='Word Cloud Visualization')

    return fig, tfidf_df

def pca_visualization_plot(df, tfidf_df):
    """
    Applies PCA on TF-IDF data to reduce dimensions and plots the results.

    Args:
    df (pandas.DataFrame): The original DataFrame.
    tfidf_df (pandas.DataFrame): DataFrame containing TF-IDF features.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly figure object showing the PCA scatter plot.
    """
    # Data preparation and PCA application
    X = tfidf_df
    y = df['food']  # Example target variable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    y = y[(X_pca[:,0] > -0.5) & (X_pca[:,0] < 1.5)]
    X_pca = X_pca[(X_pca[:,0] > -0.5) & (X_pca[:,0] < 1.5)]
    y = y[(X_pca[:,1] > -0.5) & (X_pca[:,1] < 4)]
    X_pca = X_pca[(X_pca[:,1] > -0.5) & (X_pca[:,1] < 4)]

    # Plot configuration using Plotly Express
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y)
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.update_layout(title='PCA Visualization', xaxis_title='Component 1', yaxis_title='Component 2')

    return fig

if __name__=="__main__":
    # Example usage: Load data and execute functions
    df = pd.read_csv(r'C:\Users\User\Documents\thien\Udacity\github_clone\Udacity-Project-2\data\disaster_merged_data.csv')
    genres_distribution_plot(df)
    label_distribution_plot(df)
