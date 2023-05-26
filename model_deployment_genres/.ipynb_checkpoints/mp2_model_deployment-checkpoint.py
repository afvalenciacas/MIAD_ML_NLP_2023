#!/usr/bin/python

import pandas as pd
import numpy as np
import joblib
import sys
import os
import nltk
import spacy, re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

def predict_genres(year, title, plot, rating):
    
    lr = joblib.load(os.path.dirname(__file__) + '/genres_movie_lr.pkl')
    vectorizer = joblib.load(os.path.dirname(__file__) + '/vectorizer.pkl')
    mlb = joblib.load(os.path.dirname(__file__) + '/mlb.pkl')
    
    columns = ['plot']
    data = [['plot']]
    df = pd.DataFrame(data, columns=columns)
    df.index.name = 'ID'

    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did','didnt', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves','left', 'dont', 'doesnt', 
             'im', 'hes', 'shes', 'isnt','wont','n']

    nlp = spacy.load('es_core_news_sm')
    wordnet_lemmatizer = WordNetLemmatizer()

    def preprocess_plot(plot):
        letters_only = re.sub("[^a-zA-Z]", " ", plot)
        words = letters_only.lower().split()
        stops = stopwordlist
        meaningful_words = [wordnet_lemmatizer.lemmatize(w) for w in words if not w in stops]
        cleaned_plot = " ".join(meaningful_words)
    
        doc = nlp(cleaned_plot)
        lemmas = [token.lemma_ for token in doc]
        lemmatized_plot = ' '.join(lemmas)
    
        return lemmatized_plot

    df['plot'] = df['plot'].apply(preprocess_plot)
    
    
    X = vectorizer.transform(df['plot'])
    
    #Prediccion
    predictions = lr.predict(X[0])
    
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    
    predicted_genres = pd.DataFrame(predictions, columns=cols)

    
    return predictions


if __name__ == "__main__":
    
    if len(sys.argv) < 5:
        print('Esta API necesita todos los parámetros, por favor verifica e intenta nuevamente.')
    else:
        year = sys.argv[1]
        title = sys.argv[2]
        plot = sys.argv[3]
        rating = sys.argv[4]
        
        predicted_genres = predict_genres(year, title, plot, rating)
        
        print('Género(s) de la película:', predicted_genres)