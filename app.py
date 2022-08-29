# import standard libraies
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import base64
import streamlit.components.v1 as components

# set background
# Set background title
st.set_page_config(
    page_title="Language detection app",
    layout="wide",
    initial_sidebar_state="expanded"
)


# set background image
# set the background color
@st.cache(persist=True, show_spinner=False)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: 100 cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('My project(1).png')

# read the data
data = pd.read_csv('language_detection.csv')
# Let's divided the data into X and y
X = data['Text'].values
y = data['language'].values

# And convert the text into array using the tdifvectorizer because our
tfid = TfidfVectorizer()
tfid.fit(X)
X = tfid.transform(X)

# install the model
multinomial = MultinomialNB()
# fit the train data to our model
multinomial.fit(X, y)
# let's ((predict the language detection))


def main():
    text = st.text_area("Enter your lanuge in the box", 'Enter text here')
    st.button("Predict")
    if len(text) < 1:
        st.write("")
    else:
        text = [text]
        text_int = tfid.transform(text)
        prediction = multinomial.predict(text_int)
        st.success(f"The language is  {prediction[0]}")
        st.balloons()


if __name__ == '__main__':
    main()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# with html container
with st.container():

    local_css("style.css")
    HtmlFile = open("index.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    print(source_code)
    components.html(source_code, height=600)
