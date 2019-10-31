import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ The datafiles at messages_filepath and categories_filepath
    are used as input and panda dataframe is produced as output """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """ Perform a series of data clean up activities on the dataframe
    taken as input and a cleaned dataframe is generated as output """

    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    colnames = []
    for val in row:
        colnames.append(val[:-2])
    category_colnames = colnames
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df
   

def save_data(df, database_filename):
    """ saves a dataframe to a database table """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster_Response', engine, index=False)


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