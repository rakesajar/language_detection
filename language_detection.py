# Imports
import nltk
nltk.download('punkt')

import fasttext
import pandas as pd
import streamlit as st
import plotly.express as px
from nltk.tokenize import sent_tokenize



# Class
class LanguageDetection:
    def __init__(self):
        self.language_detection_model = fasttext.load_model('language_detection_model.ftz')

    def get_prediction(self, text):
        result = self.language_detection_model.predict(str(text))
        language_code = result[0][0].split('__')[-1]
        confidence_score = result[1].item()
        return language_code, confidence_score


# Main
if __name__ == '__main__':
    ld_model = LanguageDetection()
    """
    # Language Detection App
    """
    str_text = st.text_area("Enter text to detect language:", "My name is Raj. Mein Name ist Raj. Je m'appelle Raj. I work for SIA.")
    sentences = str(str_text).split('\n')
    sentences = [sent.strip() for sentence in sentences for sent in sent_tokenize(sentence)]

    language_code, confidence_score = [], []
    for sentence in sentences:
        result = ld_model.get_prediction(sentence)
        language_code.append(result[0])
        confidence_score.append(result[1])

    """
    ### Results
    """
    df = pd.DataFrame({'Sentence': sentences, 'Language Code': language_code, 'Confidence Score': confidence_score})
    language_predicted = "mul" if df['Language Code'].nunique() > 1 else df['Language Code'].iloc[0]
    language_confidence = df['Confidence Score'].mean()
    st.write(f"**Language Predicted:** {language_predicted}")
    st.write(f"**Language Confidence:** {language_confidence:.4f}")

    count_df = df.groupby('Language Code').count().reset_index().iloc[:, :2]
    count_df.columns = ['Language Code', 'Count']
    fig = px.bar(count_df, x='Language Code', y="Count", title="Language Code Distribution")
    st.plotly_chart(fig)

    unique_languages = df['Language Code'].unique().tolist()
    language_choices = st.multiselect("Select Language Code", unique_languages, default=unique_languages)
    rdf = df[df['Language Code'].isin(language_choices)]
    st.write(rdf)
