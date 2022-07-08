# STREAMLIT

# conda activate enviro
# cd "/Users/hp/OneDrive/Documents/Python Anaconda/Streamlit_NLP_App/AppPredictionEmotion"
# streamlit run AppPredictionEmotion.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import pipeline
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sns

# other libraries
from PIL import Image
import joblib

pipe = open('models/saved_pipeline.pkl', 'rb')
rf = pickle.load(pipe)
pipe.close()

data = pd.read_csv("data/emotion_dataset.csv")

def main():
    # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
    # animations: https://lottiefiles.com/search?q=pc&category=animations
    st.set_page_config(page_icon=":computer:", layout="wide")

    st.sidebar.title("NAVIGATION")

    menu = ["Home", "Monitor", "About"]
    
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Home":
        st.subheader("Home Emotion")
    

    
    st.sidebar.markdown("""---""")

    st.sidebar.subheader("More info:"); 
    st.sidebar.write(":computer: https://jaroslavkotrba.com")

    st.sidebar.write("Copyright © 2022")


    st.markdown("""---""")

    st.subheader("More info:")
    st.write("To see other author’s projects: https://jaroslavkotrba.com")
            # ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
    st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()