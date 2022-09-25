import numpy as np

def isInKorean(input_s):
  for c in input_s:
      if ord('가') <= ord(c) <= ord('힣'):
          return 1
  return 0

def softmax(x):
  if not isinstance(x,np.ndarray):
    x = np.array(x)
  f_x = np.exp(x) / (np.sum(np.exp(x)) + 1e-9)
  return f_x