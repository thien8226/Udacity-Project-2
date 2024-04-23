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

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # get the column names
    column_names = []
    sample_data = categories.loc[0,'categories'].split(';')
    for x in sample_data:
        column_names.append(x[:-2])

     # create 36 dummy column and process one hot data
    temp_dummy_df = categories['categories'].str.split(';', expand=True)
    for column_index in temp_dummy_df.columns:
        temp_dummy_df[column_index] = temp_dummy_df[column_index].apply(lambda x: x.split('-')[-1])
    
    # create a dataframe of the 36 individual category columns
    categories[column_names] = temp_dummy_df
    df = pd.merge(messages, categories.drop('categories',axis=1), on='id')
        
    return df

def clean_data(df):
    df = df.drop_duplicates(subset=list(df.columns),keep='first')

    return df
        
def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("disaster_data_table", engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()