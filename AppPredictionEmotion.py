# STREAMLIT

# deploy to Heroku - via GitHub
# conda activate enviro
# cd "/Users/hp/OneDrive/Documents/PythonAnaconda/Streamlit_NLP_App/AppPredictionEmotion"
# streamlit run AppPredictionEmotion.py

# Track Utils
# from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table

# # Table 1
# create_page_visited_table()

# # Table 2
# create_emotionclf_table()

# Streamlit
import streamlit as st
import altair as alt
import plotly.express as px

# Basic
import numpy as np
import pandas as pd
from sklearn import pipeline
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
from datetime import datetime

# Load model
import joblib
pipe = joblib.load(open('models/saved_pipeline.pkl', 'rb'))

def predict_emotion(x):
    results = pipe.predict([x])
    return results[0]

def predict_emotions_proba(x):
    results = pipe.predict_proba([x])
    return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
    # animations: https://lottiefiles.com/search?q=pc&category=animations
    st.set_page_config(page_icon=":book:", layout="wide")

    st.sidebar.title("NAVIGATION")

    menu = ["Home", "About"] #  "Monitor"
    
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Emotion in Text")

        # add_page_visited_details("Home",datetime.now())

        with st.form(key='emotion_clf_form', clear_on_submit=True):
                raw_text = st.text_area("Type here:")
                submit_text = st.form_submit_button(label="Submit")

        if (submit_text and raw_text==""):
            st.warning("Please fill the form!")
        else: 
            if submit_text:
                left_column, right_column = st.columns(2)

                # Model
                prediction = predict_emotion(raw_text)
                probability = predict_emotions_proba(raw_text)

                # add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())

                # Left side
                with left_column:
                    st.success("Original Text")
                    st.write(f'<p style="color:#0091EA">{raw_text}</p>', unsafe_allow_html=True)

                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{} {}".format(prediction, emoji_icon))
                    st.write("Confidence: {:.4f}".format(np.max(probability)))

                # Rigth side
                with right_column:
                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probability, columns=pipe.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)

    # elif choice == "Monitor":
    #     st.subheader("Monitor App")

    #     add_page_visited_details("Monitor",datetime.now())

    #     with st.expander("Page Metrics"):
    #         page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
    #         st.dataframe(page_visited_details)	

    #         pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
    #         c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
    #         st.altair_chart(c,use_container_width=True)	

    #         p = px.pie(pg_count,values='Counts',names='Pagename')
    #         st.plotly_chart(p, use_container_width=True)

    #     with st.expander('Emotion Classifier Metrics'):
    #         df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
    #         st.dataframe(df_emotions)

    #         prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
    #         pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
    #         st.altair_chart(pc,use_container_width=True)

    elif choice == "About":
        st.subheader("About App")

        # add_page_visited_details("About",datetime.now())

        st.write("I created this app to be able to predict emotions in the given text. The main purpose is to evaluate long emails from my boss :)")

        st.markdown("""---""")

        # CONTACT
        # Use local CSS
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style.css")

        left_column, right_column = st.columns(2)
        with left_column:
            st.write("##")
            import requests
            from streamlit_lottie import st_lottie

            def load_lottieurl(url):
                r = requests.get(url)
                if r.status_code != 200:
                    return None
                return r.json()
                
            lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_naj9ijgt.json")

            st_lottie(lottie_coding, height=300, key="coding")
            
        with right_column:
            with st.container():
                st.header("Contact me: ")
                st.write("##")
                # Documention: https://formsubmit.co/
                contact_form = """
                <form action="https://formsubmit.co/jaroslav.kotrba@gmail.com" method="POST">
                    <input type="hidden" name="_captcha" value="false">
                    <input type="text" name="name" placeholder="Your name" required>
                    <input type="email" name="email" placeholder="Your email" required>
                    <textarea name="message" placeholder="Your message here" required></textarea>
                    <button type="submit">Send</button>
                </form>
                """
            st.markdown(contact_form, unsafe_allow_html=True)

    st.sidebar.markdown("""---""")

    st.sidebar.subheader("More info:"); 
    st.sidebar.write("https://jaroslavkotrba.com")

    st.markdown("""---""")

    st.subheader("More info:")
    st.write("To see other author’s projects: https://jaroslavkotrba.com")
    st.write("Copyright © 2022")
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