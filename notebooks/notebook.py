# STREAMLIT

# conda activate enviro
# cd "/Users/hp/OneDrive/Documents/Python Anaconda/AppPredictionEmotion"

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sn

import os
os. getcwd()
path = "/Users/HP/OneDrive/Documents/Python Anaconda/Streamlit_NLP_App/AppPredictionEmotion/data"
os.chdir(path)
os.listdir()

# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Load
df = pd.read_csv("emotion_dataset.csv"); df
df.columns