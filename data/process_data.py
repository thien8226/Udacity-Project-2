import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data from specified file paths, process the categories
    data into binary format and merge with messages data on 'id'.

    Parameters:
    messages_filepath (str): File path for the messages data CSV.
    categories_filepath (str): File path for the categories data CSV.

    Returns:
    pd.DataFrame: Merged DataFrame containing processed messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # get the column names from first row of categories data
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
    categories = categories.drop('categories',axis=1)

    # remove non binary data in one hot dummy columns
    for column in categories.columns[1:]:
        categories = categories[categories[column].isin([0,1])]

    df = pd.merge(messages, categories, on='id')
        
    return df

def clean_data(df):
    """
    Remove duplicate records from the dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame to be deduplicated.

    Returns:
    pd.DataFrame: Deduplicated DataFrame.
    """
    df = df.drop_duplicates(subset=list(df.columns),keep='first')

    return df
        
def save_data(df, database_filename):
    """
    Save the DataFrame to a SQLite database and a CSV file.

    Parameters:
    df (pd.DataFrame): DataFrame to be saved.
    database_filename (str): Filename for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("disaster_data_table", engine, index=False, if_exists='replace')  
    df.to_csv(r'data\disaster_merged_data.csv')


def main():
    """
    Main function to execute script functionality based on command line inputs.
    Requires three command line arguments to run properly.
    """
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
