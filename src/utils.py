import pymysql
from pymysql.constants import CLIENT

def isInKorean(input_s):
  for c in input_s:
      if ord('가') <= ord(c) <= ord('힣'):
          return 1
  return 0

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
            tag_has_music_insert_sql = """INSERT INTO tag_has_music(tag_id,music_id,tag_rank)
                                                VALUES((SELECT id FROM tag WHERE name=%s),%s,%s)"""
            tag_insert_sql = """INSERT INTO tag(name) VALUES(%s)"""
            for rank,tag in enumerate(tag_list):
                tag_search_sql = """SELECT id from tag where name=%s"""

                cursor.execute(tag_search_sql,(tag))
                if cursor.fetchone() is None:
                    cursor.execute(tag_insert_sql,(tag))

                cursor.execute(tag_has_music_insert_sql,(tag,musicid,rank+1))
        db.commit()