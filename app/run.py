import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
import joblib
from sqlalchemy import create_engine
from templates.plot_function import *

app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and lemmatize text into clean tokens.

    Parameters:
    text (str): The text to be tokenized and lemmatized.

    Returns:
    list: A list of clean tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_data_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    """
    Index route to render the main page with visualizations.

    Uses Plotly for generating visualizations from the disaster data.
    
    Returns:
    Rendered HTML template for the main page including visual data.
    """
    # extract data needed for visuals
    plot_1 = genres_distribution_plot(df)
    plot_3, tfidf_df = word_cloud_plot(df)
    plot_4 = pca_visualization_plot(df, tfidf_df)

    # create visuals
    graphs = [plot_1, plot_3, plot_4]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    """
    Route to handle user queries and display model predictions.

    Parameters:
    query (str): User input obtained from the request to classify.

    Returns:
    Rendered HTML template displaying the classification results.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    """
    Main function to run the Flask application.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
