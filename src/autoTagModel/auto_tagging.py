import torch
from src.autoTagModel.lyricAutoTag import KoSBERT,LyricAutoTagModel
from flask import Flask, jsonify,request
from flask_cors import CORS
import argparse
import json
import boto3
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=5000,help='port number to access from middleware or front')
    parser.add_argument('--thresh',type=float,default=0.12,help='threshold for keyword')
    parser.add_argument('--top_n',type=int,default=10,help='top-n for keyword')
    parser.add_argument("--model_name",type=str,default='sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    args = parser.parse_args()
    return args

app = Flask (__name__)
CORS(app)

@app.route('/music/tag/extraction',methods=["POST"])
def tag_extraction():
    lyric = request.get_json()['musicLyric']
    tag_list = auto_tag_model.get_keyword(lyric)
    return jsonify({"tagList":tag_list})

def get_data_from_s3(aws_access_key_id,aws_secret_access_key,object_name="meta_list.json",region_name="ap-northeast-2",bucket_name="music-auto-tag"):
    s3r = boto3.resource('s3',
            region_name=region_name,
            aws_access_key_id = "AKIAYVCOKHPQEOMAQPOT",
            aws_secret_access_key = "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")

    bucket = s3r.Bucket(bucket_name)
    object = bucket.Object(object_name)
    response = object.get()
    data = json.load(response['Body'])

    return data



if __name__ == "__main__":
    args = parse_args()

    assert os.environ.get("DB_HOST") is not None, "should register DB account in environment variables."
    assert os.environ.get("DB_USER") is not None, "should register DB account in environment variables."
    assert os.environ.get("DB_NAME") is not None, "should register DB account in environment variables."
    assert os.environ.get("DB_PASSWORD") is not None, "should register DB account in environment variables."

    host = os.environ.get("DB_HOST")
    user = os.environ.get("DB_USER")
    db = os.environ.get("DB_NAME")
    password = os.environ.get("DB_PASSWORD")

    assert os.environ.get("AWS_ACCESS_KEY_ID") is not None, "should register DB account in environment variables."
    assert os.environ.get("AWS_SECRET_ACCESS_KEY") is not None, "should register DB account in environment variables."

    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    data = get_data_from_s3(aws_access_key_id,aws_access_key_id,object_name="meta_list.json")

    ENDPOINT_PATH = "http://10.1.3.30:8000"
    dynamo_resource = boto3.resource('dynamodb',
                        region_name='ap-northeast-2',
                        endpoint_url = ENDPOINT_PATH,
                        aws_access_key_id = aws_access_key_id,
                        aws_secret_access_key = aws_secret_access_key)


    auto_tag_model = LyricAutoTagModel(model_name=args.model_name,top_n=args.top_n,sim_thresh=args.thresh)

    app.run(host='0.0.0.0',port=args.port)

