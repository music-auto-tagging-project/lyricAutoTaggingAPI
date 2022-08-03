import torch
from src.autoTagModel.lyricAutoTag import KoSBERT,LyricAutoTagModel
from transformers import AutoModel,AutoTokenizer
from flask import Flask, jsonify,request
from flask_cors import CORS
from google.cloud import translate_v2 as translate
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=5000,help='port number to access from middleware or front')
    parser.add_argument('--thresh',type=float,default=0.12,help='threshold for keyword')
    parser.add_argument('--top_n',type=int,default=10,help='top-n for keyword')
    args = parser.parse_args()
    return args

app = Flask (__name__)
CORS(app)

@app.route('/music/tag/extraction',methods=["POST"])
def tag_extraction():
    lyric = request.get_json()['musicLyric']
    tag_list = auto_tag_model.get_keyword(lyric)
    return jsonify({"tagList":tag_list})

if __name__ == "__main__":
    args = parse_args()

    model = KoSBERT(
        AutoModel.from_pretrained('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'),
        AutoTokenizer.from_pretrained('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'),
        torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    )
    auto_tag_model = LyricAutoTagModel(model,translator=translate.Client(),top_n=args.top_n,sim_thresh=args.thresh)

    app.run(host='0.0.0.0',port=args.port)
