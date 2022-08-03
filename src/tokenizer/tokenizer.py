import importlib

class LyricTokenizer:
  def __init__(self,name="kiwi"):
    assert name.lower() in self.tokenizer_list,f"can't find {name.lower()} tokenizer. tokenizer list : {self.tokenizer_list}"
    
    module = self.get_module(name)
    self.tokenizer = self.get_tokenizer(name,module)

  @property
  def tokenizer_list(self):
    return ["kiwi"]

  def get_module_name(self,name):
    return name.lower() + "Tokenizer"
  
  def get_class_name(self,name):
    return name.lower().capitalize() + "Tokenizer"

  def get_module(self,name):
    return importlib.import_module("src.tokenizer."+self.get_module_name(name))
    
  def get_tokenizer(self,name,module):
    return getattr(module,self.get_class_name(name))()
