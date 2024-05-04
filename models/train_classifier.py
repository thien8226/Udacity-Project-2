import sys
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
nltk.download(['punkt', 'wordnet'])

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse_Table',engine)
    category_names = list(set(df.columns)-{'id', 'message', 'original', 'genre'})
    
    X = df['message'].values
    Y = df[category_names].values
    Y = Y.astype(float)
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50], # 100
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print('classification Report for Class',category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
        
    
def save_model(model, model_filepath):
    # Save to file in the current working directory
    dump(model, model_filepath)
    model = load(model_filepath)

def main():
    import sklearn
    import numpy
    
    print('sklearn.__version__:',sklearn.__version__)
    print('numpy.__version__:',numpy.__version__)

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()