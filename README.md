# Disaster Response Pipeline

### Udacity Data Scientist Nanodegree Project

The goal of this project was building a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender. 

The model is trained on pre-labeled disaster data provided by [Figure Eight](https://www.figure-eight.com/), using CountVectorizer and TfidfTransformer for natural language processing, and MultiOutputClassifier, AdaBoostClassifier with DecisionTreeClassifier as base estimator as classifiers with GridSearch CV for hyper parameter tuning. 

The model is then deployed in a web app built with Flask and Bootstrap, where you can input a new message and get classification results in several disaster categories. The app displays also some data visualizations made with Plotly.

## 1. Usage

a) Web App

The web app is hosted on Heroku and can be tested via this [link](http://localhost:3001/).

b) Retraining the Model

To update the model, a new message dataset can be uploaded to the SQLite database with the script ´process_data.py´ in the *data* folder. The script should be run with the paths to the messages (csv file), disaster categories (csv file) and the name of the database as additional arguments:
'''
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
'''

The model can then be retrained with the script ´train_classifier.py´ in the *models* folder with the paths to the SQLite database and the classifier as additional arguments:
'''
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
'''

Warning: The ML pipeline uses GridSearchCV with very few paramters for reducing the training time, but retraining the model might still take several hours.

## 2. File Structure

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

- requirements.txt

## 3. Requirements

The requirements for this projects are listed in the ´requirements.txt´ file.

## 4. Licensing

This project is licensed under the terms of the MIT license.
