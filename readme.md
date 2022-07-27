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

## api.py
Because this api server accesses DB server, **should give some information to login DB.**

```
python api.py \
    --host "db server endpoint" \
    --user "user id to login DB" \
    --db "DB name to be accessed" \
    --password "user password to login DB" \
    -- port(optional) "port number(default 5000)" \
```

