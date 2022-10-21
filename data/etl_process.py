import sys
import pandas as pd
from sqlalchemy import create_engine
from langdetect import detect

def load_data(messages_filepath, categories_filepath):
    """Load and Merge messages and categories data
    
    Arguments:
        messages_filepath {str} -- CSV filepath of message data.
        categories_filepath {str} -- CSV filepath of categories data.
    
    Returns:
        Dataframe -- Pandas Dataframe of mereged data
    """
    # Load data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge data
    df = messages.merge(categories, how='outer',on='id')
    
    
    return df


def clean_data(df):
    """Feature engineering and Data cleaning.
    1. Prepare target Data for multi-output classifier
    2. Fix data type
    3. Drop duplicates and unnecessary data
    
    Arguments:
        df {Dateframe} -- Pandas Dataframe of mereged data
    
    Returns:
        df -- Pandas Dataframe of processed data
    """
    # Prepare target data for multi-output classifier
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0,:]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].transform(lambda x: x[-1:])
        # Fix Data type
        categories[column] = categories[column].astype(int)
    df = df.drop('categories',axis=1)
    df = pd.concat([df, categories], axis = 1)

    # Drop unnecessary data
    df.drop('child_alone',axis=1, inplace = True)
    df = df[df['related'] != 2]
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """Save processed data to a SQLite database
    
    Arguments:
        df {Dataframe} -- Pandas Dataframe of processed data
        database_filename {str} -- Table name
    """
    database_filename
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # Save data to df table
    df.to_sql('df', engine, index=False)  


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