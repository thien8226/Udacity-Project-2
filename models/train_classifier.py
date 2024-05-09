import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download(['punkt', 'wordnet'])
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

def load_data(database_filepath):
    """
    Load data from the SQLite database and extract features and labels for model training.
    
    Parameters:
    database_filepath (str): File path to the SQLite database.

    Returns:
    tuple: Feature data (X), label data (Y), and category names for the labels.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_data_table', engine)
    category_names = list(set(df.columns) - {'id', 'message', 'original', 'genre'})
    
    X = df['message'].values
    Y = df[category_names].values
    Y = Y.astype(float)
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and lemmatize text to produce clean tokens.

    Parameters:
    text (str): Text to be tokenized and lemmatized.

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

def build_model():
    """
    Build a machine learning pipeline and set up grid search for hyperparameter tuning.

    Returns:
    GridSearchCV: Grid search object with pipeline and parameter grid.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50]  # Example to adjust n_estimators
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance on the test set and print classification reports for each category.

    Parameters:
    model: Trained model to evaluate.
    X_test: Test features.
    Y_test: Test labels.
    category_names: Names of the categories.
    """
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print('Classification Report for Class:', category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
    
def save_model(model, model_filepath):
    """
    Save the trained model to a specified filepath.

    Parameters:
    model: Model to be saved.
    model_filepath (str): Filepath to save the model.
    """
    # Save to file in the current working directory
    dump(model, model_filepath)
    model = load(model_filepath)

def main():
    import sklearn
    import numpy
    
    print('sklearn.__version__:', sklearn.__version__)
    print('numpy.__version__:', numpy.__version__)

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE:', database_filepath)
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL:', model_filepath)
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
