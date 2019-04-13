import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    Load and split the data from a specified database filepath
    Inputs: database_filepath - the user specified path to the data
    Outputs: X - The messages
             Y - The categories of those messages
             category_names - the corresponding names of those categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    category_names = df.iloc[:, 5:].columns
    X = df.message.values
    Y = df.iloc[:,5:].values
    return X, Y, category_names

def tokenize(text):
    '''
    Takes a text string, tokenizes, lemmatizes and case-normalizes it
    Inputs: text - a text string
    Outputs: clean_tokens - a list of cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds a model based on parameters researched earlier. This is the result of an exhaustive 5 hour grid search.
    Inputs: none
    Outputs: pipeline - An sklearn pipeline model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
    'tfidf__use_idf': [True, False],
    'vect__max_df': [0.5, 0.75],
    'vect__max_features': [None, 5000],
    'clf__estimator__max_depth': [5,10],
    'clf__estimator__n_estimators' : [50, 100, 200],
    'clf__estimator__min_samples_split': [2, 3, 4],
    'clf__estimator__class_weight':['balanced']
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 2, scoring = make_scorer(f1_score, average = 'weighted'))
    
    cv.fit(X_train, Y_train)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evalutes the model using sklearns 'classification report' function to return precision and recall for each class
    Inputs: model - an sklearn model (or pipeline)
            X_test - the test set data
            Y_test - the correct categories for the test data
            category_names - the names of the categories being predicted
    '''
    Y_pred = model.predict(X_test)
    for i in range(Y_pred.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Saves an sklearn model to a pickle file
    Inputs: model - an sklearn trained model
            model_filepath - the user specified location for the model
    Outputs: None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Takes in user arguments for database input and pickle model output to load data and train a model
    Prints to terminal the evaluation of that model    
    Finishes by saving that model to a pickle file to be used in the web app
    '''
    
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