from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nnsplit import NNSplit
# from nltk.corpus import stopwords
import re
import pandas as pd

def get_polarity(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    compound = score['compound']
    return compound

def number_words(sentence):
    return len(re.findall(r'\w+', str(sentence)))


def perform_vader_classification(new_review):
    # Replace with new input
    # new_review = "You When I booked with your company on line you showed me pictures of a room I thought I was getting and paying for and then when we arrived that s room was booked and the staff told me we could only book the villa suite theough them directly Which was completely false advertising After being there we realised that you have grouped lots of rooms on the photos together leaving me the consumer confused and extreamly disgruntled especially as its my my wife s 40th birthday present Please make your website more clear through pricing and photos as again I didn t really know what I was paying for and how much it had wnded up being Your photos told me I was getting something I wasn t Not happy and won t be using you again "

    sent_list = []
    splitter = NNSplit("en")

    sent = splitter.split([new_review])
    for i in sent[0]:
        new_string = ''
        for j in i:
            new_string += j.text + " "
        sent_list.append(new_string)
        
    sent_list_lower = [sent.lower() for sent in sent_list]

    # stop_list = stopwords.words('english')
    # sent_list_lower_no_stopword_list = [[word for word in sent.split() if not word in stop_list] for sent in sent_list_lower]
    # sent_list_lower_no_stopword = []
    # for sent in sent_list_lower_no_stopword_list:
    #     new_sent = ' '.join(sent)
    #     sent_list_lower_no_stopword.append(new_sent)

    data = pd.DataFrame(sent_list_lower, columns=["sentence"])
    data['polarity'] = data['sentence'].apply(get_polarity)

    length = (data['sentence'].apply(number_words) >= 8)
    data = data.loc[length]

    polarity = data['polarity'].mean()
    if polarity >= 0.05:
        sentiment = ('positive', polarity)
    elif polarity > -0.05 and polarity < 0.05: 
        sentiment = ('neutral', polarity)
    else: 
        sentiment = ('negative', polarity)
    

    # sentiment is the aggregated sentiment for the entire review
    # data is the table that will store the individual sentence details (polarity)
    # so still must think of a way to return both vader-processed data and aggregated sentiment per review
    return sentiment 

# def assign_sentiment(reviews_df):
