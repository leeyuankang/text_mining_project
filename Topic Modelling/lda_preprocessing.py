# TEST NEW DATA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora

lemmatizer= WordNetLemmatizer()

stop_list = stopwords.words('english')
stop_list += ['hotel', 'know', 'i', 'have', 'would', 'take', 'a', 'choose', 'the', 'first', 'second', 'lovely', 'will', 'definitely', 'longer', 'stayed', 'also']

def preprocessing(review):
    sentences = review.split(". ")
    data = [[word.lower() for word in x.split() if word.lower() not in stop_list] for x in sentences]
    lem = [[lemmatizer.lemmatize(w) for w in doc] for doc in data]
    dict_lem=corpora.Dictionary(lem)
    token_to_id2=dict_lem.token2id
    vec_lem= [dict_lem.doc2bow(doc) for doc in data]
    
    return vec_lem

unseen_rev= preprocessing("their service is good. But the paper is bad.")
                          
for sen in unseen_rev:
#     tester_model is the lda model that you load with pickle (rmb to change the path)
    tester_model= pickle.load(open("/Users/jaslynwong/TextAnalytics/Project/ldamallet_model.pickle",'rb'))
    vector=tester_model[sen]
    vector = sorted(vector, key=lambda x: x[1], reverse=True)
    topic = vector[0][0]
    print(topic)