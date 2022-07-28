import torch
from src.model import KoSBERT,LyricAutoTagModel
from src.utils import get_lyric_by_musicDB,save_tag_list_in_db
from transformers import AutoModel,AutoTokenizer
from flask import Flask, jsonify
from flask_cors import CORS
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",type=str,help='db server endpoint')
    parser.add_argument("--user",type=str,help='db login id')
    parser.add_argument("--db",type=str,help='db name')
    parser.add_argument("--password",type=str,help='db login password')
    parser.add_argument('--port',type=int,default=5000,help='port number to access from middleware or front')
    args = parser.parse_args()
    return args

app = Flask (__name__)
CORS(app)

@app.route('/tag/<int:musicid>',methods=["GET"])
def tag_extraction(musicid):
    lyric = get_lyric_by_musicDB(musicid,host,user,db,password)
    tag_list = auto_tag_model.get_keyword(lyric)
    save_tag_list_in_db(tag_list,musicid,host,user,db,password)
    return jsonify({"tagList":tag_list})

if __name__ == "__main__":
    args = parse_args()

    host,user,db,password = args.host, args.user, args.db, args.password

    model = KoSBERT(
        AutoModel.from_pretrained('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'),
        AutoTokenizer.from_pretrained('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'),
        torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    )
    auto_tag_model = LyricAutoTagModel(model)

    app.run(host='0.0.0.0',port=args.port)
