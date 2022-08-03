from abc import *

class BaseAutoTag(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractclassmethod
    def get_keyword(self):
        pass