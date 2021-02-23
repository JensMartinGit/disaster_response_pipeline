import sys, re
import pandas as pd
from sqlalchemy import engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nlt.download('wordnet')
from nltk.tokenize import word_tokenize


def load_data(database_filepath):
	'''Load the clean data from sqlite database and assign X, y and category_names
	
	Arguments:
		database_filepath: filepath to sqlite database
	Returns:
		X: pandas.Series, messages to classify
		y: pandas.DataFrame, classification into categories as one-hot encoding
		category_names: list, category names
	'''

    # Load data from sqlite database into DataFrame
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM messages', engine)
    
    # Assign X, y and category names
    X = df.messages
    y = df.iloc[:,4:]
    category_names = list(y.columns)
    
    return X, y, category_names


def tokenize(text):
    '''Process text data into tokens

    Replaces all urls by a placeholder, word tokenizes, lemmatizes, and normalizes the text, 
    and returns a list of tokens.

    Arguments:
    	text: string, input text
    Returns:
    	token_list: list, tokenized words
    '''

    # Replace URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'url_placeholder', text)

    # Word tokenize text
    token_list = word_tokenize(text)

    # Lemmatize and normalize tokens
    token_list = [WordNetLemmatizer().lemmatize(w).lower().strip() for w in token_list]

    return token_list



def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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