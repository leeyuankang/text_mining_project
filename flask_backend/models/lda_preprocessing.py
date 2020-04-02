# TEST NEW DATA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora
import gensim
import os
import pickle

lemmatizer= WordNetLemmatizer()

stop_list = stopwords.words('english')
stop_list += ['e', 'etc','u','hotel','always', 'know', 'have', 'would', 'take', 'choose', 'the', 'first', 'second', 'lovely', 'will', 'definitely', 'longer', 'stayed', 'also']

pickle_file_dir = "/Users/soonhangchye/Desktop/text_mining_project/flask_backend/models/ldamallet_model5050783.pickle"
tester_model= pickle.load(open(pickle_file_dir,'rb'))

gensim_lda = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(tester_model)

def preprocessing(review):
    sentences = review.split(". ")
    data = [[word.lower() for word in x.split() if word.lower() not in stop_list] for x in sentences]
    words_only= [[w for w in doc if re.search('^[a-z]+$', w)] for doc in data]
    
    lem = [[lemmatizer.lemmatize(w) for w in doc] for doc in words_only]
    dict_lem=corpora.Dictionary(lem)
    token_to_id2=dict_lem.token2id
    vec_lem= [dict_lem.doc2bow(doc) for doc in lem]
    
    return vec_lem

# unseen_rev= preprocessing("their service is good. But the paper is bad.")
# print(unseen_rev)

topic_names={0: 'general', 1: 'room comfort',2: 'food',3: 'reception service',4: 'room appliance',5: 'price',6: 'overall stay experience',7: 'accessibility',8: 'room noises',9: 'staff',10: 'room size upgradable',11: 'location amenities',12: 'hotel amenities',13: 'room lighting and temperature',14: 'booking'}


def assign_topic(sentence):
    
    sentence = preprocessing(sentence)

    #     tester_model is the lda model that you load with pickle (rmb to change the path)
    vector=gensim_lda[sentence]
    vector=vector[0]
    vector = sorted(vector, key=lambda x: x[1], reverse=True)
    topic = vector[0][0]
    return topic_names[topic]

