import pickle
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import pandas as pd, numpy as np


stop_list = stopwords.words('english')
file_dir = '/Users/soonhangchye/Desktop/text_mining_project/flask_backend/models/classification.pkl'

with open(file_dir, 'rb') as file:
    logistic = pickle.load(file)[0]
    
with open(file_dir, 'rb') as file:
    user_uni = pickle.load(file)[1]

def stem(array):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in array]

def lemmetize(array):
    lemmatizer = WordNetLemmatizer() 
    return [lemmatizer.lemmatize(w) for w in array]

def label_review(review):
    # need to convert to dataframe level
    # remove stopwords
    review = review.split()
    review = [word for word in review if word not in stop_list]

    # make words case-insensitive
    review = [word.lower() for word in review]

    # remove punctuations if any
    review = [re.sub('[^\w\s]','', word) for word in review]

    # stemming with NLTK
    review = stem(review)

    # turn arrays for each row in df['']'
    review = " ".join(review)
    review = pd.Series(review)

    review = user_uni.transform(review)

    prediction = logistic.predict(review)

    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"


