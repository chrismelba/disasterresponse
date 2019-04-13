# Disaster Response Pipeline Project
This project is designed to classify text into a category to help with disaster response. It will take a text input and determine if it falls into a category (or multiple categories).

When you run this project you will have a web browser where you can type your own text and try it out, along with some interesting extra graphs to explore the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
