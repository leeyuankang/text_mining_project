from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd, numpy as np
import pickle

import requests, json

from models.vader_classification import assign_sentiment
# from models.doc_classification import label_review
from models.lda_preprocessing import assign_topic

upload_folder = "./upload_folder"

app = Flask(__name__)

CORS(app)

@app.route("/reviews_analysis/file_upload", methods=['POST'])
def process_file_upload():

    ## only use the following chunk if you have multiple files and intend to save a copy of them
    # for key, f in request.files.items():
    #     if key.startswith('file'):
    #         f.save(os.path.join(upload_folder, f.filename))

    # retrieve the uploaded file from frontend
    uploaded_file = request.files['file']

    # convert CSV into dataframe before running vader and topic modelling function
    reviews_df = pd.read_csv(uploaded_file)

    # insert row count (no of reviews)
    topbar_data = [reviews_df.shape[0]]
    
    reviews_df['review_id'] = reviews_df.index + 1

    reviews_df = reviews_df.reindex(columns=['review_id','reviews'])

    # label the reviews into positive and negative
    # but cannot find the pickle file some reason
    # reviews_df['Sentiment(Classification)'] = reviews_df['reviews'].apply(label_review)

    # assign sentiment and polarity in sentence level using vader
    reviews_df = assign_sentiment(reviews_df)
    
    # assign topic to each sentence using lda
    reviews_df['topic'] = reviews_df['sentence'].apply(assign_topic)

    # create dictionary for first data table (sentence level analysis)
    sen_lvl_data = sentence_lvl_analysis_data(reviews_df)

    # append sentiment distribution
    topbar_data.append(get_sentiment_dist(reviews_df))   

    # append topic count into topbar
    topbar_data.append(reviews_df['topic'].nunique())  

    # create chart data
    chart_data = prepare_chart_data(reviews_df)

    result = {'topbar_data': topbar_data, 'chart_data': chart_data, 'sen_lvl_data': sen_lvl_data}

    result = jsonify(result)

    return result, 201

def sentence_lvl_analysis_data(reviews_df):
    reviews_df = reviews_df[['review_id', "sentence", "topic", 'sen_lvl_polarity', 'sen_lvl_sentiment']]
    reviews_df = reviews_df.rename(columns={'sen_lvl_polarity': 'polarity', 'sen_lvl_sentiment': 'sentiment'})
    reviews_df = reviews_df.reindex(columns=['review_id', 'sentence', 'topic', 'polarity', 'sentiment'])
    return reviews_df.to_dict('records')

def get_sentiment_dist(reviews_df):
    pos_count = reviews_df[reviews_df['sen_lvl_sentiment'] == 'Positive'].shape[0]
    neg_count = reviews_df[reviews_df['sen_lvl_sentiment'] == 'Negative'].shape[0]
    neu_count = reviews_df[reviews_df['sen_lvl_sentiment'] == 'Neutral'].shape[0]

    total = pos_count + neg_count + neu_count

    return [round(pos_count/total*100), round(neu_count/total*100), round(neg_count/total*100)]

def prepare_chart_data(reviews_df):
    chart_df = reviews_df.groupby('topic')['sen_lvl_sentiment'].value_counts()
    # chart_df = reviews_df.groupby('topic')['sen_lvl_sentiment'].value_counts(normalize=True)

    chart_df = chart_df.unstack(level = -1)
    # chart_df = chart_df.unstack(level = -1)

    chart_df = chart_df.reset_index()
    # chart_df = chart_df.reset_index()

    chart_df = chart_df.round({'Positive':0, 'Negative':0, 'Neutral':0})
    # chart_df = chart_df.round({'Positive':2, 'Negative':2, 'Neutral':2})
    
    chart_df = chart_df.set_index('topic')
    # chart_df = chart_df.set_index('topic')

    chart_df = chart_df.fillna(0)
    # chart_df = chart_df.fillna(0)

    result = chart_df.to_dict('index')

    
    topics = list(result.keys())
    
    pos_list = [percentage['Positive'] for percentage in result.values()]
    neg_list = [percentage['Negative'] for percentage in result.values()]
    neutral_list = [percentage['Neutral'] for percentage in result.values()]

    chart_data = {
        "topics": topics,
        "positive": pos_list,
        "negative": neg_list,
        "neutral": neutral_list
    }
    print(chart_data)
    return chart_data

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 8001, debug = True)