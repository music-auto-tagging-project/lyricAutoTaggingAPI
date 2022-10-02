from typing import List

import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel,AutoTokenizer

from src.utils import isInKorean,softmax
from src.autoTagModel.base import BaseAutoTag
from src.tokenizer.tokenizer import LyricTokenizer

class LyricAutoTagModel(BaseAutoTag):
  def __init__(self,
              model_name='sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens',
              target_pos_list=["NN","NNG"],
              general_label_list = ["따뜻한","차가운","슬픈","밝은","신나는","우울한"],
              tokenizer='kiwi',
              top_n=5,sim_thresh=0.12,max_chunk_length=256,n_gram_range=(1,1)):
    self.model = KoSBERT(
            AutoModel.from_pretrained(model_name),
            AutoTokenizer.from_pretrained(model_name),
            torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
    self.top_n = top_n
    self.sim_thresh = sim_thresh
    self.max_chunk_length = max_chunk_length
    self.target_pos_list = target_pos_list
    self.n_gram_range = n_gram_range
    self.lyric_tokenizer = LyricTokenizer(name=tokenizer).tokenizer
    self.general_label_list = general_label_list = np.array(general_label_list)

    self.general_label_embeddings = self.model.encode(general_label_list.tolist())

  def get_keyword(self,lyric):
    assert isInKorean(lyric),"The lyric should include Korean."

    lyric_chunk = self.get_lyric_chunk(lyric)
    candidates = self.get_lyric_keyword_candidate(lyric)

    doc_embedding = self.model.encode(lyric_chunk)
    mean_doc_embedding = np.average(doc_embedding,axis=0)
    if not isinstance(candidates,List):
      candidates = candidates.tolist()

    candidate_embeddings = self.model.encode(candidates)

    distances = cosine_similarity(mean_doc_embedding[np.newaxis,:], candidate_embeddings).flatten()
    k = min(distances.shape[0],self.top_n)
    indices = distances.argsort()[:-k:-1]

    keywords=[]
    keyword_embs=[]
    keyword_sims=[]
    for index in indices:
      if distances[index] < self.sim_thresh:
        break
      if len(candidates[index]) >1 and len(candidates[index]) < 5:
        keywords.append(candidates[index])
        keyword_embs.append(candidate_embeddings[index].tolist())
        keyword_sims.append(distances[index].item())

    sim_weights = softmax(distances.flatten())
    mood_embedding = np.stack(candidate_embeddings,axis=0)
    mood_embedding = np.average(mood_embedding,axis=0,weights=sim_weights)

    general_label_sim = cosine_similarity(mood_embedding.reshape(1,-1),self.general_label_embeddings)
    mood_keywords = self.general_label_list[np.flip(np.argsort(general_label_sim))].tolist()

    results={
      "lyric_emb":mean_doc_embedding.tolist(),
      "keywords":keywords,
      "mood_keywords":mood_keywords,
      "keyword_embs":keyword_embs,
      "keyword_sims":keyword_sims
    }

    return results

  def get_lyric_chunk(self,lyric):
    chunk_list=[]
    sentences = self.lyric_tokenizer.split_sentence(lyric)

    is_first = True
    chunk = ""
    for sen in sentences:
      if is_first:
        chunk = sen.replace("\n"," ")
        is_first = False
        continue

      if len(chunk) + len(sen) < self.max_chunk_length:
        chunk += (" " + sen.replace("\n"," "))
      else:
        chunk_list.append(chunk.strip())
        chunk=""
        is_first = True
    
    if chunk:
      chunk_list.append(chunk.strip())

    return chunk_list

  def get_lyric_keyword_candidate(self,lyric):
    lyric_pos_list = []

    word_pos_list = self.lyric_tokenizer.tokenize(lyric)
    for word,pos in word_pos_list:
      if len(word)<=1 and len(word)>=5:
        continue
      for target_pos in self.target_pos_list:
        if pos in target_pos:
          lyric_pos_list.append(word)
          break
    
    lyric_doc = ' '.join(lyric_pos_list)
    
    assert lyric_doc is not None,f"lyric doc is empty."

    count = CountVectorizer(ngram_range=self.n_gram_range).fit([lyric_doc]) # tokenized_word에서 ngram_range의 단어들을 counting
    candidates = count.get_feature_names_out() # fit된 단어 중 중복이 제거된 단어 list
      
    return candidates

class KoSBERT(nn.Module):
  def __init__(self,model,tokenizer,device):
    super(KoSBERT,self).__init__()
    self.tokenizer = tokenizer
    self.device = device
    self.model = model.to(device)

  @torch.no_grad()
  def encode(self,x):
    self.model.eval()
    device = self.device
    encoded_x = self.tokenizer(x,padding=True,truncation=True,return_tensors='pt').to(device)
    out = self.model(**encoded_x)
    sentence_embeddings = self.mean_pooling(out, encoded_x['attention_mask'])
    self.model.train()
    return sentence_embeddings.detach().cpu().numpy()
  
  def forward(self,x):
    device = self.device
    encoded_x = self.tokenizer(x,padding=True,truncation=True,return_tensors='pt').to(device)
    return self.model(**encoded_x)
    
  #Mean Pooling - Take attention mask into account for correct averaging
  def mean_pooling(self,model_output, attention_mask):
      token_embeddings = model_output[0] #First element of model_output contains all token embeddings
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)