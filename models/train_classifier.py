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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    category_names = df.iloc[:, 5:].columns
    X = df.message.values
    Y = df.iloc[:,5:].values
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
        ('vect', CountVectorizer(tokenizer=tokenize, max_df = 0.5, max_features = 5000)),
        ('tfidf', TfidfTransformer(use_idf = False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth = 10, min_samples_split = 3, n_estimators = 200, class_weight = 'balanced')))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(Y_pred.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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