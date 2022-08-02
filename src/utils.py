import pymysql
from pymysql.constants import CLIENT
import re

def isInKorean(input_s):
  for c in input_s:
      if ord('가') <= ord(c) <= ord('힣'):
          return 1
  return 0

def eng_sent_tokenize(lyric):
  lyric = re.sub(r"\b(\w+)(\-)(\w+)\b",r"\1\3",lyric)
  sents=[]
  for sent in lyric.split("\r\n"):
    for s in sent.split("\n"):
      if not s:
        continue
      sents.append(s)
  return sents

def get_lyric_by_musicDB(musicid,host,user,db,password):
    db = pymysql.connect(host=host,user=user,db=db,password=password,charset='utf8',client_flag=CLIENT.MULTI_STATEMENTS)
    with db:
        with db.cursor() as cursor:
            sql = """SELECT lyric FROM music WHERE id=%s"""
            cursor.execute(sql,(musicid))
            lyric = cursor.fetchone()[0]
    return lyric

def save_tag_list_in_db(tag_list,musicid,host,user,db,password):
    db = pymysql.connect(host=host,user=user,db=db,password=password,charset='utf8',client_flag=CLIENT.MULTI_STATEMENTS)
    with db:
        with db.cursor() as cursor:
            delete_prev_tag_sql = """DELETE FROM tag_has_music WHERE music_id=%s"""
            cursor.execute(delete_prev_tag_sql,(musicid))
            
            tag_has_music_insert_sql = """INSERT INTO tag_has_music(tag_id,music_id,tag_rank)
                                                VALUES((SELECT id FROM tag WHERE name=%s),%s,%s)"""
            tag_insert_sql = """INSERT IGNORE INTO tag(name) VALUES(%s)"""
            for rank,tag in enumerate(tag_list):
                cursor.execute(tag_insert_sql,(tag))
                cursor.execute(tag_has_music_insert_sql,(tag,musicid,rank+1))
        db.commit()