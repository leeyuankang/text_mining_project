# TEST NEW DATA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora
import gensim
import os
import pickle

lemmatizer= WordNetLemmatizer()

stop_list = stopwords.words('english')
stop_list += ['hotel', 'know', 'i', 'have', 'would', 'take', 'a', 'choose', 'the', 'first', 'second', 'lovely', 'will', 'definitely', 'longer', 'stayed', 'also']

pickle_file_dir = "/Users/soonhangchye/Desktop/text_mining_project/flask_backend/models/ldamallet_model478927.pickle"
tester_model= pickle.load(open(pickle_file_dir,'rb'))

gensim_lda = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(tester_model)

def preprocessing(review):
    sentences = review.split(". ")
    data = [[word.lower() for word in x.split() if word.lower() not in stop_list] for x in sentences]
    lem = [[lemmatizer.lemmatize(w) for w in doc] for doc in data]
    dict_lem=corpora.Dictionary(lem)
    token_to_id2=dict_lem.token2id
    vec_lem= [dict_lem.doc2bow(doc) for doc in data]
    
    return vec_lem

unseen_rev= preprocessing("their service is good. But the paper is bad.")
# print(unseen_rev)

def assign_topic(sentence):
    
    sentence = preprocessing(sentence)

    #     tester_model is the lda model that you load with pickle (rmb to change the path)
    vector=gensim_lda[sentence]
    #     print(vector)
    vector = sorted(vector, key=lambda x: x[1], reverse=True)
    topic = vector[0][0][0]
    return topic