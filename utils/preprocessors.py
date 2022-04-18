# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import os
import numpy as np

from utils.common import load_json, write_json, word_convert, is_punct
from utils.common import PAD, UNK, NUM
from utils.common import ROOT_WORD, ROOT_POS_TAG

glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6),
               '2B': int(1.2e6)}


class Preprocessor(object):

  def __init__(self, config):
    self.config = config

  @staticmethod
  def load_dataset(filename, keep_number=False, lowercase=True):
    dataset = []
    for record in load_json(filename):
      words = [word_convert(word,
                            keep_number=keep_number,
                            lowercase=lowercase)
               for word in record["words"]]
      sent = {"sent_id": record["sent_id"],
              "words": [ROOT_WORD] + words,
              "pos_tags": [ROOT_POS_TAG] + record["pos_tags"],
              "heads": record["heads"],
              "labels": record["labels"]}
      if "train_sent_ids" in record:
        sent["train_sent_ids"] = record["train_sent_ids"]
      dataset.append(sent)
    return dataset

  @staticmethod
  def load_vocab(path):
    print("Loading word embeddings from %s" % path)
    f = codecs.open(path, mode='r', encoding='utf-8',
                    errors='backslashreplace')
    if "glove" not in path:
      _ = f.readline()
    return [line.lstrip().rstrip().split(" ")[0] for line in f]

  @staticmethod
  def build_char_vocab(datasets):
    chars = set()
    for dataset in datasets:
      for record in dataset:
        for word in record["words"]:
          for char in word:
            chars.add(char)
    char_vocab = [PAD, UNK] + sorted(list(chars))
    return dict([(char, idx) for idx, char in enumerate(char_vocab)])

  @staticmethod
  def build_vocab(datasets, field, include_unk=False):
    vocab = set()
    for dataset in datasets:
      for record in dataset:
        for elem in record[field]:
          vocab.add(elem)
    vocab = sorted(list(vocab))
    if include_unk:
      vocab = vocab + [UNK]
    return dict([(elem, idx) for idx, elem in enumerate(vocab)])

  @staticmethod
  def build_word_vocab_pretrained(vocab):
    word_vocab = [PAD, UNK, NUM, ROOT_WORD] + sorted(vocab)
    return dict([(word, idx) for idx, word in enumerate(word_vocab)])

  @staticmethod
  def filter_emb(word_dict, path):
    print("Creating word embeddings...")
    f = codecs.open(path, mode='r', encoding='utf-8',
                    errors='backslashreplace')
    if "glove" in path:
      dim = len(f.readline().lstrip().rstrip().split(" ")[1:])
    else:
      dim = int(f.readline().split()[-1])
    f.close()

    n_tokens = 0
    vectors = np.zeros([len(word_dict) - 4, dim])
    f = codecs.open(path, mode='r', encoding='utf-8',
                    errors='backslashreplace')
    if "glove" not in path:
      # Discard the first line of fastText
      f.readline()
    for line in f:
      n_tokens += 1
      if n_tokens % 10000 == 0:
        print("%d" % n_tokens, flush=True, end=" ")
      line = line.lstrip().rstrip().split(" ")
      word_index = word_dict[line[0]] - 4
      vector = [float(v) for v in line[1:]]
      vectors[word_index] = np.asarray(vector)
    f.close()
    return vectors

  @staticmethod
  def build_dataset(data, word_dict, char_dict, pos_tag_dict, label_dict,
                    add_heads=True, add_labels=True):
    dataset = []
    for record in data:
      chars_ids = []
      words_ids = []
      punct_ids = []
      for i, word in enumerate(record["words"]):
        chars = [char_dict[char]
                 if char in char_dict else char_dict[UNK] for char in word]
        chars_ids.append(chars)
        if i == 0:
          word_id = word_dict[ROOT_WORD]
        else:
          punct_ids.append(int(is_punct(word)))
          word = word_convert(word, keep_number=False, lowercase=True)
          word_id = word_dict[word] if word in word_dict else word_dict[UNK]
        words_ids.append(word_id)
      pos_tags_ids = [pos_tag_dict[tag]
                      if tag in pos_tag_dict else pos_tag_dict[UNK]
                      for tag in record["pos_tags"]]
      sent = {"sent_id": record["sent_id"],
              "words": words_ids,
              "chars": chars_ids,
              "puncts": punct_ids,
              "pos_tags": pos_tags_ids}
      if add_heads:
        sent["heads"] = record["heads"]
      if add_labels:
        labels_ids = [label_dict[label] if label in label_dict else -1
                      for label in record["labels"]]
        sent["labels"] = labels_ids
      if "train_sent_ids" in record:
        sent["train_sent_ids"] = record["train_sent_ids"]
      dataset.append(sent)
    return dataset

  def preprocess(self):
    config = self.config
    os.makedirs(config["save_path"], exist_ok=True)

    train_data = self.load_dataset(
      os.path.join(config["raw_path"], "train.json"),
      keep_number=False, lowercase=True)
    train_data = train_data[:config["data_size"]]

    # build token vocabulary
    if self.config["use_words"] or self.config["use_bert"] is False:
      emb_path = self.config["emb_path"]
      emb_vocab = self.load_vocab(emb_path)
      word_dict = self.build_word_vocab_pretrained(emb_vocab)
      vectors = self.filter_emb(word_dict, emb_path)
      np.savez_compressed(config["pretrained_emb"], embeddings=vectors)
    else:
      word_dict = {}

    # build tag & label dicts
    pos_tag_dict = self.build_vocab(
      datasets=[train_data], field="pos_tags", include_unk=True)
    label_dict = self.build_vocab(
      datasets=[train_data], field="labels", include_unk=False)
    print("\nLoading train & valid sets")

    # load dataset
    train_data = self.load_dataset(
      os.path.join(config["raw_path"], "train.json"),
      keep_number=True, lowercase=config["char_lowercase"])
    valid_data = self.load_dataset(
      os.path.join(config["raw_path"], "valid.json"),
      keep_number=True, lowercase=config["char_lowercase"])

    # build character dict
    train_data = train_data[:config["data_size"]]
    valid_data = valid_data[:config["data_size"]]
    char_dict = self.build_char_vocab([train_data])

    print("Converting train & valid sets into indices")
    # create indices dataset
    train_set = self.build_dataset(
      train_data, word_dict, char_dict, pos_tag_dict, label_dict)
    valid_set = self.build_dataset(
      valid_data, word_dict, char_dict, pos_tag_dict, label_dict)
    vocab = {
      "word_dict": word_dict, "char_dict": char_dict,
      "pos_tag_dict": pos_tag_dict, "label_dict": label_dict}

    print("Train Sents: %d" % len(train_set))
    print("Valid Sents: %d" % len(valid_set))

    # write to file
    write_json(os.path.join(config["save_path"], "vocab.json"), vocab)
    write_json(os.path.join(config["save_path"], "train.json"), train_set)
    write_json(os.path.join(config["save_path"], "valid.json"), valid_set)
    print("Preprocessing completed!!\n")
