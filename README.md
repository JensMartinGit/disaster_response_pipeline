# Disaster Response Pipeline

### Udacity Data Scientist Nanodegree Project

The goal of this project was building a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender. 

The model is trained on pre-labeled disaster data provided by [Figure Eight](https://www.figure-eight.com/), using **CountVectorizer** and **TfidfTransformer** for natural language processing, and **MultiOutputClassifier** combined with an **AdaBoostClassifier** with **DecisionTreeClassifier** as base estimator as classifiers, with **GridSearchCV** for hyper parameter tuning. 

The model is then deployed in a web app built with **Flask** and **Bootstrap**, where you can input a new message and get classification results in several disaster categories. The app displays also some data visualizations made with **Plotly**.

## 1. Usage

a) Web App

The web app can be run with the script `run.py` in the *app* folder.

b) Retraining the Model

To update the model, a new message dataset can be uploaded to the SQLite database with the script `process_data.py` in the *data* folder. The script should be run with the paths to the messages (csv file), disaster categories (csv file) and the name of the database as additional arguments:
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

The model can then be retrained with the script `train_classifier.py` in the *models* folder with the paths to the SQLite database and the classifier as additional arguments:
```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

**Warning:** The ML pipeline uses GridSearchCV with very few paramters for reducing the training time, but retraining the model might still take several hours.

## 2. File Structure

* app
	* template
    * master.html (# main page of web app)
    * go.html  (# classification result page of web app)
	* run.py  (# Flask file that runs app)  
    
* data
    * disaster_categories.csv  (# data to process) 
    * disaster_messages.csv  (# data to process)
    * process_data.py (# runs ETL pipeline and stores clean data in SQLite database)
    * InsertDatabaseName.db   (# database to save clean data to) 

* models
    * train_classifier.py (# runs ML pipeline and saves trained model in .pkl file)
    * classifier.pkl  (# saved model) 

* README.md  

* requirements.txt

## 3. Requirements

The requirements for this projects are listed in the `requirements.txt` file.

## 4. Discussion of Results

The classification report below shows that the results of the model are far from perfect and leave lots of room for improvement:

There are several possible explanations for this. The **tuning of the model's hyper parameters** was done with GridSearchCv, but due to hardware restrictions I implemented the model with a really small parameter grid in order to reduce the training time. For a real world application it would definitively be neccessary to run GridSearchCV with additional parameters, which probably would lead to better evaluation scores.

Furthermore, one could ask if the **choice of algorithms** used here (MultiOutputClassifier combined with AdaBoostClassifier with DecisionTreeClassifier as base estimator) really was the best decision for this kind of classification task. However, other classification algorithms (which I tried with smaller data sets due to long training times) didn't perform significantly better.

But the most important reason for the imperfect performance of the model becomes clear when looking at the distribution of the 36 different categories in the training data set:

The visualization reveals how imbalanced the training data set really is, which inevitably leads to **imbalance bias** in the model. For one category, *Child alone*, there is not one single data record in the training data, and for several other categories there are little more than 100 data points, compared to almost 20,000 for the *related* category. And, as the classification report shows, the categories with the bad evaluation scores are usually the ones with very few data points, while for e.g. the *related* category both precision and recall scores are quite good.

The important question is if one could rework the training data set to handle this imbalance. There are three common approaches for this, but all are problematic in this case. **Undersampling**, i.e. sampling from the majority class to keep only a part of these data points, would lead to an extremely reduced training data set here. **Oversampling**, i.e. replicating some points from the minority class, or creating new synthetic data points with methods like **SMOTE** could be an option, but both methods have their downsides, and are especially problematic when used on categories with so few data points like here.

A better option here would be to either collect more balanced training data or create a new classification scheme with less and more balanced categories, I think.

## 5. Licensing

This project is licensed under the terms of the MIT license.
