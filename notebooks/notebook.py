# STREAMLIT

# conda activate enviro

import numpy as np
import pandas as pd
from sklearn import pipeline
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sns

import os
os. getcwd()
path = "/Users/HP/Dropbox/Documents/Python/AppPredictionEmotion"
os.chdir(path)
os.listdir()

# Load
df = pd.read_csv("data/emotion_dataset_raw.csv"); df

df = df[['Text', 'Emotion']]

# Value counts
df['Emotion'].value_counts()
sns.countplot(x='Emotion', data=df);

# Text Cleaning Pkgs
import neattext.functions as nfx

# User handels (@#)
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

# Stopwords
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

# Special characters
# df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_special_characters)

# Feature & labels
X = df['Clean_Text']; X
y = df['Emotion']; y

# Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Build pipeline
from sklearn.pipeline import Pipeline

# Models
from sklearn.feature_extraction.text import CountVectorizer # matrix of word counts

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

pipe = Pipeline(steps=[('cv', CountVectorizer()), ('lr', MultinomialNB())])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

# Accuracy Naive Bayes Classifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

pipe.score(X_test, y_test)

print('Confusion metrix: ', '\n', confusion_matrix(y_test, y_pred))
print('Accuracy: ', round(accuracy_score(y_test, y_pred)*100, 2),'%')
print('Report: ', '\n', classification_report(y_test, y_pred))

# Logistic Regression
from sklearn.linear_model import LogisticRegression

pipe = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

# Accuracy Logistic Regression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

pipe.score(X_test, y_test)

print('Confusion metrix: ', '\n', confusion_matrix(y_test, y_pred))
print('Accuracy: ', round(accuracy_score(y_test, y_pred)*100, 2),'%')
print('Report: ', '\n', classification_report(y_test, y_pred))

# Example
example = "This book was interesting it made me happy!"
pipe.predict([example])[0]
pipe.predict_proba([example])
pipe.classes_

# Saving the model
import joblib
saved_pipeline = open("models/saved_pipeline.pkl", "wb")
joblib.dump(pipe, saved_pipeline)
saved_pipeline.close()




