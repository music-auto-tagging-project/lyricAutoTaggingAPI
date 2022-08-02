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