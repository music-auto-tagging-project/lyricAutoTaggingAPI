from abc import *

class BaseTokenizer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractclassmethod
    def tokenize(self):
        pass

    @abstractclassmethod
    def split_sentence(self):
        pass
