# STREAMLIT

# conda activate enviro
# cd "/Users/hp/OneDrive/Documents/Python Anaconda/AppPredictionEmotion"

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sns

import os
os. getcwd()
path = "/Users/HP/OneDrive/Documents/Python Anaconda/Streamlit_NLP_App/AppPredictionEmotion"
os.chdir(path)
os.listdir()

# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Text Cleaning Pkgs
import neattext.functions as nfx

# Load
df = pd.read_csv("data/emotion_dataset.csv"); df

# Value counts
df['Emotion'].value_counts()
sns.countplot(x='Emotion', data=df);

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


