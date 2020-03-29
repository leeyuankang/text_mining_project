from flask import Flask, request
from flask_cors import CORS
import os

import requests, json

from models.vader_classification.py import 

upload_folder = "./upload_folder"

app = Flask(__name__)

CORS(app)

@app.route("/reviews_analysis/file_upload", methods=['POST'])
def process_file_upload():
    for key, f in request.files.items():
        if key.startswith('file'):
            f.save(os.path.join(upload_folder, f.filename))

    return "Upload success", 201

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 8001, debug = True)