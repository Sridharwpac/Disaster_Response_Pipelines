# Disaster_Response_Pipelines

## Table Of Contents
  1. Introduction
  2. File Descriptions
  3. Results
  4. Licensing, Authors, and Acknowledgements
  5. Instructions
 
## Introduction:
This project is about analysing messages sent by people during disasters. I have applied data engineering, natural language processing, and machine learning techniques to analyze these messages to create a model for an API that classifies disaster messages. These messages could potentially be sent to appropriate disaster relief agencies.

## File Descriptions
There are three main folders:

1. data
    - disaster_categories.csv - dataset including all the categories
    - disaster_messages.csv - dataset including all the messages
    - process_data.py - ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db - output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    - train_classifier.py - machine learning pipeline scripts to train and export a classifier
    - classifier.pkl - output of the machine learning pipeline, i.e. a trained classifer
3. app
    - run.py - Flask file to run the web application
    - templates - contains html file for the web applicatin
    
## Results:

1. An ETL pipleline has been built to read data from two csv files, clean data, and save data into a SQLite database.
2. A machine learning pipepline has been developed to train a classifier to perform multi-output classification on the 36 categories available in the dataset.
3. A Flask app has been created to show data visualization and classify the message that user enters on the web page.

## Licensing, Authors, Acknowledgements:

Credits to be given to FigureEight for provding the data used by this project.

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

  - To run ETL pipeline that cleans data and stores in database python 
    - data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  - To run ML pipeline that trains classifier and saves 
    - python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app. 
    - python run.py
    
3. The web app should now be running if there were no errors. Now, open another Terminal Window and type env|grep WORK to get the SPACEID and SPACEDOMAIN
    - go to https://SPACEID-3001.SPACEDOMAIN (Do not forget to replace SPACEID and SPACEDOMAIN)
