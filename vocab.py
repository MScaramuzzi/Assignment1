from collections import OrderedDict
from copy import deepcopy
import gensim
import numpy as np
from gensim.models import KeyedVectors


# Define the class vocabulary to handle the encoding of words into token ids to give as input to the embedding layer

class Vocabulary:


  def __init__(self):
    self.word2id = OrderedDict() 
    self.id2word = OrderedDict()
    self.oov_token = 'UNK' 
    self.oov_index = 1
    self.current_index = 2  # 0 is reserved for padding, 1 for oov words

  def import_from_glove(self, embedding_model: gensim.models.keyedvectors.KeyedVectors) -> None:
    # update the token ids with the embeddings taken from GloVe
    self.word2id.update({w: (i+2) for w, i in embedding_model.key_to_index.items()})  # skip the first two indexes because they are padding and oov tags
    self.id2word.update(OrderedDict(enumerate(embedding_model.index_to_key, 2)))
    self.current_index = len(self.id2word) + 2

  def add_from_df(self, sentences: list[list[str]]) -> None:
    # add the oov words from the train set into the vocabulary 
    for sentence in sentences:
      for word in sentence:
        if word not in self.word2id:
          self.word2id[word] = self.current_index
          self.id2word[self.current_index] = word
          self.current_index += 1

  def get_OOV_words(self, sentences: list[list[str]]) -> list[str]:
    set_words = set([word for sentence in sentences for word in sentence])
    oov_words = set_words.difference(set(self.id2word.values()))
    return list(oov_words)

  def encode(self, sentences: list[list[str]]) -> list[list[int]]:
    # take all the words in the dataset and transform them into token ids

    all_sents_encoded = []
    for sentence in sentences:
      sent_encoded = []
      for word in sentence:
        if word in self.word2id:
          sent_encoded.append(self.word2id[word])
        else:
          sent_encoded.append(self.oov_index)
      all_sents_encoded.append(sent_encoded)
    return all_sents_encoded

  def decode(self, index_ids_seq: list[list[int]]) -> list[list[str]]:
  # decode the token ids to words

    all_sents_decoded = []
    for index_ids in index_ids_seq:
      sent_decoded = []
      for index_id in index_ids:
        if index_id in self.id2word:
          sent_decoded.append(self.id2word[index_id])
        else:
          sent_decoded.append(self.oov_token)
      all_sents_decoded.append(sent_decoded)
    return all_sents_decoded

  def copy(self):
    return deepcopy(self)