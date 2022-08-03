from .base import BaseTokenizer
from kiwipiepy import Kiwi

class KiwiTokenizer(BaseTokenizer):
    def __init__(self):
        self.kiwi = Kiwi()
    
    def tokenize(self,lyric):
        return [(posed.form,posed.tag) for posed in self.kiwi.tokenize(lyric)]
    
    def split_sentence(self,lyric):
        return [sent.text for sent in self.kiwi.split_into_sents(lyric)]
