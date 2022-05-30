# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from collections import defaultdict
import random

import h5py
import numpy as np

from utils.common import load_json


class Batcher(object):

  def __init__(self, config):
    self.config = config

  @staticmethod
  def pad_sequences(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
      # 0: "PAD" for words and chars, "O" for tags
      pad_tok = 0
    if max_length is None:
      max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
      seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
      sequence_padded.append(seq_)
      sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length

  def pad_char_sequences(self, sequences, max_length=None,
                         max_token_length=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
      max_length = max(map(lambda x: len(x), sequences))
    if max_token_length is None:
      max_token_length = max(
        [max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
      sp, sl = self.pad_sequences(seq, max_length=max_token_length)
      sequence_padded.append(sp)
      sequence_length.append(sl)
    sequence_padded, _ = self.pad_sequences(
      sequence_padded, pad_tok=[0] * max_token_length, max_length=max_length)
    sequence_length, _ = self.pad_sequences(
      sequence_length, max_length=max_length)
    return sequence_padded, sequence_length

  def pad_bert_reps(self, batch):
    batch_size = len(batch["seq_len"])
    max_length = max(batch["seq_len"]) - 1
    padded_words = np.zeros(
      shape=(batch_size, max_length, self.config["bert_dim"]),
      dtype=np.float32)
    for i, (reps, seq_len) in enumerate(zip(batch["bert_rep"],
                                            batch["seq_len"])):
      padded_words[i][:seq_len - 1] += reps
    batch["bert_rep"] = padded_words
    return batch

  @staticmethod
  def make_head_one_hots(batch_heads):
    n_tokens = len(batch_heads[0])
    return [np.identity(n_tokens + 1)[heads] for heads in batch_heads]

  def make_each_batch(self, **kwargs):
    raise NotImplementedError

  def batchnize_dataset(self, **kwargs):
    raise NotImplementedError


class WeightBatcher(Batcher):

  def make_each_batch(self, batch):
    batch["batch_size"] = len(batch["words"])
    batch["words"], batch["seq_len"] = self.pad_sequences(batch["words"])
    if self.config["use_chars"]:
      batch["chars"], _ = self.pad_char_sequences(batch["chars"],
                                                  max_token_length=20)
    if self.config["use_pos_tags"]:
      batch["pos_tags"], _ = self.pad_sequences(batch["pos_tags"])
    return batch

  def batchnize_dataset(self, data, batch_size, shuffle=True):
    dataset = load_json(data)

    if shuffle:
      random.shuffle(dataset)
      dataset.sort(key=lambda record: len(record["words"]))

    batches = []
    batch = defaultdict(list)
    prev_seq_len = len(dataset[0]["words"])

    for record in dataset:
      seq_len = len(record["words"])

      if len(batch["words"]) == batch_size or prev_seq_len != seq_len:
        batches.append(self.make_each_batch(batch))
        batch = defaultdict(list)

      for field in ["sent_id", "words", "chars", "pos_tags",
                    "puncts", "heads", "labels"]:
        batch[field].append(record[field])
      prev_seq_len = seq_len

    if len(batch["words"]) > 0:
      batches.append(self.make_each_batch(batch))

    if shuffle:
      random.shuffle(batches)
    for batch in batches:
      yield batch


class BERTWeightBatcher(WeightBatcher):

  def __init__(self, config):
    super(BERTWeightBatcher, self).__init__(config)
    self.train_bert_hdf5 = h5py.File(config["train_bert_hdf5"], "r")
    self.valid_bert_hdf5 = h5py.File(config["valid_bert_hdf5"], "r")

  def load_and_add_bert_reps(self, batch, use_train_bert_hdf5):
    if use_train_bert_hdf5:
      bert_hdf5 = self.train_bert_hdf5
    else:
      bert_hdf5 = self.valid_bert_hdf5
    batch["bert_rep"] = [bert_hdf5[str(sent_id)][1:]
                         for sent_id in batch["sent_id"]]
    return batch

  def make_each_batch(self, batch):
    batch["words"], batch["seq_len"] = self.pad_sequences(batch["words"])
    if self.config["use_chars"]:
      batch["chars"], _ = self.pad_char_sequences(batch["chars"],
                                                  max_token_length=20)
    if self.config["use_pos_tags"]:
      batch["pos_tags"], _ = self.pad_sequences(batch["pos_tags"])
    batch["batch_size"] = len(batch["words"])
    return batch

  def batchnize_dataset(self, data, batch_size, shuffle=True):
    dataset = load_json(data)

    if shuffle:
      random.shuffle(dataset)
      dataset.sort(key=lambda record: len(record["chars"]))

    batches = []
    batch = defaultdict(list)
    prev_seq_len = len(dataset[0]["chars"])

    for record in dataset:
      seq_len = len(record["chars"])

      if len(batch["chars"]) == batch_size or prev_seq_len != seq_len:
        batches.append(self.make_each_batch(batch))
        batch = defaultdict(list)

      for field in ["sent_id", "words", "chars", "pos_tags",
                    "puncts", "heads", "labels"]:
        batch[field].append(record[field])
      prev_seq_len = seq_len

    if len(batch["chars"]) > 0:
      batches.append(self.make_each_batch(batch))

    if shuffle:
      random.shuffle(batches)
    for batch in batches:
      yield batch
