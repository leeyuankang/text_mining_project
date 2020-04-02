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

def det_sentiment(polarity):
    if polarity >= 0.05:
        return 'Positive'
    elif polarity > -0.05 and polarity < 0.05: 
        return 'Neutral'
    else: 
        return 'Negative'

def perform_vader_classification(review_id, review):
    # Replace with new input
    # new_review = "You When I booked with your company on line you showed me pictures of a room I thought I was getting and paying for and then when we arrived that s room was booked and the staff told me we could only book the villa suite theough them directly Which was completely false advertising After being there we realised that you have grouped lots of rooms on the photos together leaving me the consumer confused and extreamly disgruntled especially as its my my wife s 40th birthday present Please make your website more clear through pricing and photos as again I didn t really know what I was paying for and how much it had wnded up being Your photos told me I was getting something I wasn t Not happy and won t be using you again "

    sent_list = []
    splitter = NNSplit("en")

    sent = splitter.split([review])
    for i in sent[0]:
        new_string = ''
        for j in i:
            new_string += j.text + " "
        sent_list.append(new_string)
        
    sent_list_lower = [sent.lower() for sent in sent_list]

    data = pd.DataFrame(sent_list_lower, columns=["sentence"])
    data['review_id'] = review_id
    data['sen_lvl_polarity'] = data['sentence'].apply(get_polarity)
    data['sen_lvl_sentiment'] = data['sen_lvl_polarity'].apply(det_sentiment)
    length = (data['sentence'].apply(number_words) >= 8)
    data = data.loc[length]
    data = data.reindex(columns=['review_id','sentence', 'sen_lvl_polarity', 'sen_lvl_sentiment'])

    # review level polarity
    data['review_lvl_polarity'] = data['sen_lvl_polarity'].mean()
    
    data['review_lvl_sentiment'] = data['review_lvl_polarity'].apply(det_sentiment)
 
    return data
    # return [review_id, review, polarity, sentiment]

    # sentiment is the aggregated sentiment for the entire review
    # data is the table that will store the individual sentence details (polarity)
    # so still must think of a way to return both vader-processed data and aggregated sentiment per review
    # return sentiment 

def assign_sentiment(reviews_df):
    # dataframe columns: 
    # review_id, sentence, topic, sen_lvl_polarity, sen_lvl_sentiment, rev_lvl_polarity, rev_lvl_sentiment

    # create an empty dataframe with required headers first
    processed_reviews_df = pd.DataFrame(columns=['review_id', 'sentence', 'sen_lvl_polarity', 'sen_lvl_sentiment', 'review_lvl_polarity', 'review_lvl_sentiment'])

    # iterate through reviews_df with .loc
    for i in range(len(reviews_df)):
        temp_df = perform_vader_classification(reviews_df.loc[i,"review_id"], reviews_df.loc[i,"reviews"])
        processed_reviews_df = pd.concat([processed_reviews_df,temp_df], axis=0, ignore_index=True)

    return processed_reviews_df