Tweet Sentiment Analysis NLP
Overview

This project implements a sentiment analysis system to classify tweets into positive or negative sentiment using Natural Language Processing techniques and machine learning models.
The objective is to build an end-to-end NLP pipeline that preprocesses raw tweet text, extracts meaningful features, trains classification models, and evaluates their performance using appropriate metrics.

Dataset
The dataset consists of tweets labeled with sentiment classes. Each record contains:
Tweet text
Sentiment label (binary classification)
The dataset is used to train and evaluate sentiment classification models.

Methodology
Text Preprocessing

The following preprocessing steps are applied:
Conversion of text to lowercase
Removal of URLs and user mentions
Normalization of extra whitespace
Feature Engineering
TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical feature vectors
Unigram and bigram representations are applied to capture contextual information

Modeling

The following machine learning models are implemented:
Logistic Regression
Support Vector Machine (SVM)
Model selection is based on performance comparison across evaluation metrics.

Evaluation

Models are evaluated using:
Accuracy
F1-score
F1-score is emphasized to ensure a balanced evaluation of precision and recall.
Technologies Used
Python
Pandas
NumPy
Scikit-learn

Project Structure
Tweet-sentiment-analysis-nlp/
│
├── sentiment_analysis.ipynb
├── README.md
├── requirements.txt
└── data/

How to Run:
Clone the repository:
git clone https://github.com/parul14346/Tweet-sentiment-analysis-nlp.git
Navigate to the project directory:
cd Tweet-sentiment-analysis-nlp
Install dependencies:
pip install -r requirements.txt
Run the Jupyter notebook:
Open sentiment_analysis.ipynb
Execute all cells sequentially
Results
The final model demonstrates effective performance on validation data and is capable of capturing sentiment patterns present in tweet text.
