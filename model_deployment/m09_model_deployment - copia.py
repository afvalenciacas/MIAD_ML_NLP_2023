#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import nltk
import spacy, re
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('es_core_news_sm')
nltk.download('stopwords')
nltk.download('wordnet')

def predict_genres(year, title, plot, rating):

    lr = joblib.load(os.path.dirname(__file__) + '/genres_movie_lr.pkl')
    
    columns = ['plot']
    data = [['plot']]
    df = pd.DataFrame(data, columns=columns)
    df.index.name = 'ID'

    wordnet_lemmatizer = WordNetLemmatizer()

    def clean_plot(plot):      
        letters_only = re.sub("[^a-zA-Z]", " ", plot) 
        words = letters_only.lower().split()                             
        stops = set(stopwords.words("english"))   
        meaningful_words = [wordnet_lemmatizer.lemmatize(w) for w in words if not w in stops]   
        return( " ".join( meaningful_words ))

    df['plot'] = df['plot'].apply(clean_plot)
    
    def lemmatize_text(text):
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
        return ' '.join(lemmas)

    df['plot'] = df['plot'].apply(lemmatize_text)
    
    vectorizer = TfidfVectorizer(max_features = 3250, stop_words ='english', smooth_idf=True, use_idf= True, sublinear_tf=True, norm='l1', analyzer='word', strip_accents='unicode')
    X = vectorizer.fit_transform(df['plot'])
    
    #Prediccion
    p1 = lr.predict(X[0])

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) < 5:
        print('This API needs all parameters, please check and try again')
        
    else:

        year = sys.argv[1]
        title = sys.argv[2]
        plot = sys.argv[3]
        rating = sys.argv[4]
        
        p1 = predict(plot)
        
        print('Genero o generos de la película: ', p1)
        