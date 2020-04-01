from flask import Flask, request
from flask_cors import CORS
import pandas as pd, numpy as np
import pickle

import requests, json

from models.vader_classification import assign_sentiment

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

    # reviews_df = assign_sentiment(reviews_df)

    return "Upload success", 201


def label_by_review_level(review_df):
    classification_model = pickle.load(open('./models/classification.pkl','rb'))

    

    return "Hello"

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 8001, debug = True)