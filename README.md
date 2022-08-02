# LyricAutoTag API server
this api server provides lyric auto tagging model by musicid in DB

## requirements
```
transformers==4.20.1
kiwipiepy==0.13.1
pymysql==1.0.2
flask
```

`pip install -r requirements.txt`

## auto_tag_api.py
```
python api.py \
    -- treshhold "keyword should be upper then this threshold."
    -- top_n "maximum keyword number"
    -- port "port number(default 5000)" \
```

- operating api server in background : `python api.py ~~ %`
- terminating api server in background : `sh terminate_api.sh`