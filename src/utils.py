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
