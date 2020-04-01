import pickle
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd, numpy as np

def stem(array):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in array]

def lemmetize(array):
    lemmatizer = WordNetLemmatizer() 
    return [lemmatizer.lemmatize(w) for w in array]

def label_review(review):
    stop_list = stopwords.words('english')

    # need to convert to dataframe level
    # remove stopwords
    review = [word for word in review if word not in stop_list]
