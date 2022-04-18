# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import gzip
import re
import pickle
import ujson
import unicodedata

import numpy as np

PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
NULL = "<NULL>"
ROOT_WORD = "_ROOT_"
ROOT_POS_TAG = "ROOT"
ROOT_HEAD = -1
ROOT_LABEL = "root"


def load_json(filename):
  with codecs.open(filename, mode='r', encoding='utf-8') as f:
    dataset = ujson.load(f)
  return dataset


def write_json(filename, data):
  with codecs.open(filename, mode="w", encoding="utf-8") as f:
    ujson.dump(data, f, ensure_ascii=False)


def load_pickle(filename):
  with gzip.open(filename, 'rb') as gf:
    return pickle.load(gf)


def write_pickle(filename, data):
  with gzip.open(filename + '.pkl.gz', 'wb') as gf:
    pickle.dump(data, gf, pickle.HIGHEST_PROTOCOL)


def word_convert(word, keep_number=True, lowercase=True):
  if not keep_number:
    if is_digit(word):
      return NUM
  if lowercase:
    word = word.lower()
  return word


def is_punct(word):
  return all(unicodedata.category(char).startswith('P') for char in word)


def is_digit(word):
  try:
    float(word)
    return True
  except ValueError:
    pass
  try:
    unicodedata.numeric(word)
    return True
  except (TypeError, ValueError):
    pass
  result = re.compile(r'^[-+]?[0-9]+,[0-9]+$').match(word)
  if result:
    return True
  return False


def l2_normalize(v):
  return v / np.maximum(np.linalg.norm(v, ord=2), 1e-10)


def summarize_vectors(vectors, sim, transform_vector="mean"):
  if sim == "cos":
    vectors = [l2_normalize(vec) for vec in vectors]
  if transform_vector == "mean":
    return np.mean(vectors, axis=0)
  return np.sum(vectors, axis=0)


def get_vocab_size_and_dim(path):
  emb = np.load(path)["embeddings"]
  vocab_size = len(emb)
  emb_dim = len(emb[0])
  print("Word Embeddings: size=%d, dim=%d" % (vocab_size, emb_dim))
  return vocab_size, emb_dim
