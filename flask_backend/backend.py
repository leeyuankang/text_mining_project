from flask import Flask, request
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

    # create chart data
    chart_data = prepare_chart_data(reviews_df)

    result = {
            'chart_data': chart_data,
            'sen_lvl_data': sen_lvl_data
        }

    return result, 201

def sentence_lvl_analysis_data(reviews_df):
    reviews_df = reviews_df[['review_id', "sentence", "topic", 'sen_lvl_polarity', 'sen_lvl_sentiment']]
    reviews_df = reviews_df.rename(columns={'sen_lvl_polarity': 'polarity', 'sen_lvl_sentiment': 'sentiment'})
    reviews_df = reviews_df.reindex(columns=['review_id', 'sentence', 'topic', 'polarity', 'sentiment'])
    print(reviews_df.columns.values)

    return reviews_df.to_dict('records')

def prepare_chart_data(reviews_df):
    chart_df = reviews_df.groupby('topic')['sen_lvl_sentiment'].value_counts(normalize=True)

    chart_df = chart_df.unstack(level = -1)
    chart_df = chart_df.reset_index()
    chart_df = chart_df.round({'Positive':2, 'Negative':2, 'Neutral':2})
    chart_df = chart_df.set_index('topic')

    # result = {
    #     0: {'Positive': 0.23, 'Negative':0.52, 'Neutral':0.14},
    #     1: {'Positive': 0.23, 'Negative':0.52, 'Neutral':0.14},
    #     2: {'Positive': 0.23, 'Negative':0.52, 'Neutral':0.14},
    # }

    result = chart_df.to_dict('index')
    # print(result)
    topics = list(result.keys())
    # print(topics)
    pos_list = [percentage['Positive'] for percentage in result.values()]
    neg_list = [percentage['Negative'] for percentage in result.values()]
    neutral_list = [percentage['Neutral'] for percentage in result.values()]
    
    chart_data = {
        "topics": topics,
        "positive": pos_list,
        "negative": neg_list,
        "neutral": neutral_list
    }

    return chart_data

def label_by_review_level(review_df):

    return "Hello"

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 8001, debug = True)