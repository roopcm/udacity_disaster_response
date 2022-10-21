import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,make_scorer
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings
warnings.simplefilter('ignore')


def load_data(database_filepath):
    """Load processed data
    
    Arguments:
        database_filepath {str} -- The filepath of SQLite database
    
    Returns:
        Dataframe -- Pandas Dataframe of processed data
    """
    # Load data
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM df", engine)
    # Split data to feature and target
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

    return X,Y,list(Y.columns.values)

def tokenize(text):
    """Tokenizer for NLP pipeline
    
    Arguments:
        text {str} -- single unprocessed message
    
    Returns:
        list -- List of processed and tokenized words
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z]", " ", text.lower()) 
    # Tokenize 
    tokens = word_tokenize(text)
    # Stem text
    stemmed = [PorterStemmer().stem(word) for word in tokens if word not in stopwords.words("english")]
    return stemmed
    
def mid_f1(y_true, y_pred):
    """Calculate average F1 score
    
    Arguments:
        y_true {dataframe} -- Pandas Dataframe actual target
        y_pred {dataframe} -- Pandas Dataframe for predict target
    
    Returns:
        [float] -- Median of F1 score
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_list.append(f1)
        
    score = np.median(f1_list)
    return score

def build_model():
    """Build a machine learning pipeline
    
    Returns:
        object -- Grid search object
    """
    # Setup seed
    np.random.seed(233)
    # Build data pipline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=False)))
    ])
    parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 25], 
              'clf__estimator__min_samples_split':[5, 10]}
    scorer = make_scorer(mid_f1)
    cv = GridSearchCV(pipeline, param_grid = parameters,scoring = scorer,verbose = 10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Calculate and display the evaluation of ML model
    
    Arguments:
        model {object} -- Grid search object
        X_test {dataframe} -- Pandas Dataframe for features
        Y_test {dataframe} -- Pandas Dataframe for targets
        category_names {list} -- list of column names of targets
    """
    # Get result
    target_names = category_names
    y_pred = model.predict(X_test)
    y_true = np.array(Y_test)

    # Build a dataframe to store metrics and scores
    df = pd.DataFrame()
    for i,target in enumerate(target_names):
        accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        precision = precision_score(y_true[:, i], y_pred[:, i])
        recall = recall_score(y_true[:, i], y_pred[:, i])
        df = df.append({'index':target,'Accuracy':accuracy,'F1 Score':f1,'Precision':precision,'Recall':recall},ignore_index = True)
    # print results
    print(df)
    print(df.describe())

def save_model(model, model_filepath):
    """Save model to the pickle file

    Arguments:
        model {object} -- Fitted model object
        model_filepath {str} -- Filepath for pickle file
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
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