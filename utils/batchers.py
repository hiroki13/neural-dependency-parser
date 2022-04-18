# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from collections import defaultdict
import random

import h5py
import numpy as np

from utils.common import load_json, summarize_vectors


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
    if "sigmoid" in self.config["model_name"]:
      batch["head_one_hots"] = self.make_head_one_hots(batch["heads"])
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


class UnlabeledInstanceBatcher(Batcher):

  def __init__(self, config):
    super(UnlabeledInstanceBatcher, self).__init__(config)
    # self.train_sents = load_json(config["train_set"])
    self.train_sents = dict([(sent["sent_id"], sent)
                            for sent in load_json(config["train_set"])])
    self.train_sent_ids = list(self.train_sents.keys())
    self.train_edge_reps_hdf5 = None

  def get_precomputed_train_rep(self, k=1, transform_vector="sum"):
    sampled_edge_reps = []
    for sent_id in random.sample(self.train_sent_ids,
                                 k=min(k, len(self.train_sents))):
      sampled_edge_reps += self.train_edge_reps_hdf5[str(sent_id)]
    return summarize_vectors(
      sampled_edge_reps, self.config["sim"], transform_vector)

  def _add_random_sents_to_batch(self, batch):
    random.shuffle(self.train_sent_ids)
    for sent_id in self.train_sent_ids[:self.config["k"]]:
      train_sent_record = self.train_sents[sent_id]
      batch["words"].append(train_sent_record["words"])
      batch["neighbor_heads"].append(train_sent_record["heads"])
      if self.config["use_chars"]:
        batch["chars"].append(train_sent_record["chars"])
      if self.config["use_pos_tags"]:
        batch["pos_tags"].append(train_sent_record["pos_tags"])
    return batch

  def _add_neighbor_sents_to_batch(self, batch):
    for sent_id in batch["train_sent_ids"][:self.config["k"]]:
      train_sent_record = self.train_sents[sent_id]
      batch["words"].append(train_sent_record["words"])
      batch["neighbor_heads"].append(train_sent_record["heads"])
      if self.config["use_chars"]:
        batch["chars"].append(train_sent_record["chars"])
      if self.config["use_pos_tags"]:
        batch["pos_tags"].append(train_sent_record["pos_tags"])
    return batch

  @staticmethod
  def _flatten(batch_neighbor_heads, max_len):
    new_deps = []
    new_heads = []
    for batch_index, heads in enumerate(batch_neighbor_heads):
      step_size = batch_index * max_len
      dep_indices = np.asarray(range(len(heads)))
      head_indices = np.asarray(heads)
      new_deps.extend(dep_indices + step_size + 1)
      new_heads.extend(head_indices + step_size)
    return new_deps, new_heads

  def make_each_batch(self, batch, add_neighbor_sents=False):
    batch["n_sents"] = len(batch["words"])

    if add_neighbor_sents:
      if "train_sent_ids" in batch:
        batch = self._add_neighbor_sents_to_batch(batch)
      else:
        batch = self._add_random_sents_to_batch(batch)

    batch["words"], batch["seq_len"] = self.pad_sequences(batch["words"])
    if self.config["use_chars"]:
      batch["chars"], _ = self.pad_char_sequences(batch["chars"],
                                                  max_token_length=20)
    if self.config["use_pos_tags"]:
      batch["pos_tags"], _ = self.pad_sequences(batch["pos_tags"])

    if add_neighbor_sents:
      batch["neighbor_deps"], batch["neighbor_heads"] = \
        self._flatten(batch["neighbor_heads"], max_len=max(batch["seq_len"]))

    batch["batch_size"] = len(batch["words"])
    return batch

  def batchnize_dataset(self, data, batch_size, shuffle=True,
                        add_neighbor_sents=False):
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
        batches.append(self.make_each_batch(batch, add_neighbor_sents))
        batch = defaultdict(list)

      for field in ["sent_id", "words", "chars", "pos_tags",
                    "puncts", "heads", "labels"]:
        batch[field].append(record[field])
      prev_seq_len = seq_len

    if len(batch["words"]) > 0:
      batches.append(self.make_each_batch(batch, add_neighbor_sents))

    if shuffle:
      random.shuffle(batches)
    for batch in batches:
      yield batch


class LabeledInstanceBatcher(Batcher):

  def __init__(self, config):
    super(LabeledInstanceBatcher, self).__init__(config)
    self.train_sents = load_json(config["train_set"])
    self.label_vocab_size = len(load_json(config["vocab"])["label_dict"])
    self.n_each_label = [0 for _ in range(self.label_vocab_size)]
    self.label_wise_sent_bank = self._make_sent_bank()
    self.label_wise_edge_reps_hdf5 = None

  def _make_sent_bank(self):
    label_wise_sent_bank = [[] for _ in range(self.label_vocab_size)]
    for record in self.train_sents:
      sent_id = record["sent_id"]
      for token_index, label in enumerate(record["labels"]):
        label_wise_sent_bank[label].append([sent_id, token_index])
    for label, sent_bank in enumerate(label_wise_sent_bank):
      n_labels = len(sent_bank)
      if n_labels == 0:
        print("ERROR: the label set %d is empty" % label)
        exit()
      else:
        self.n_each_label[label] = n_labels
    return label_wise_sent_bank

  def make_neighbor_batch_with_all_labels(self, k=1):
    batch = {"words": [], "chars": [], "pos_tags": [],
             "class_deps": [], "class_heads": [], "class_labels": []}
    for label, sent_bank in enumerate(self.label_wise_sent_bank):
      sampled_instances = random.sample(sent_bank, k=min(k, len(sent_bank)))
      for sent_id, token_index in sampled_instances:
        train_sent_record = self.train_sents[sent_id]
        batch["words"].append(train_sent_record["words"])
        if self.config["use_chars"]:
          batch["chars"].append(train_sent_record["chars"])
        if self.config["use_pos_tags"]:
          batch["pos_tags"].append(train_sent_record["pos_tags"])
        batch["class_deps"].append(token_index)
        batch["class_heads"].append(train_sent_record["heads"][token_index])
        batch["class_labels"].append(train_sent_record["labels"][token_index])
    batch["batch_size"] = len(batch["words"])
    batch["words"], batch["seq_len"] = self.pad_sequences(batch["words"])
    if self.config["use_chars"]:
      batch["chars"], _ = self.pad_char_sequences(batch["chars"],
                                                  max_token_length=20)
    if self.config["use_pos_tags"]:
      batch["pos_tags"], _ = self.pad_sequences(batch["pos_tags"])
    return batch

  def get_precomputed_label_wise_reps(self, k=1, transform_vector="sum"):
    sampled_edge_reps = []
    for label, sent_bank in enumerate(self.label_wise_sent_bank):
      sampled_instances = random.sample(sent_bank, k=min(k, len(sent_bank)))
      reps = [self.label_wise_edge_reps_hdf5[str(sent_id)][token_index]
              for sent_id, token_index in sampled_instances]
      sampled_edge_reps.append(
        summarize_vectors(reps, self.config["sim"], transform_vector))
    assert len(sampled_edge_reps) == self.label_vocab_size
    return sampled_edge_reps

  def _add_label_wise_instances_to_batch(self, batch):
    for label, sent_bank in enumerate(self.label_wise_sent_bank):
      index = np.random.randint(0, len(sent_bank))
      sent_id, token_index = sent_bank[index]
      train_sent_record = self.train_sents[sent_id]
      batch["words"].append(train_sent_record["words"])
      if self.config["use_chars"]:
        batch["chars"].append(train_sent_record["chars"])
      if self.config["use_pos_tags"]:
        batch["pos_tags"].append(train_sent_record["pos_tags"])
      batch["class_deps"].append(token_index)
      batch["class_heads"].append(train_sent_record["heads"][token_index])
    return batch

  def make_each_batch(self, batch, add_label_wise_instances=False):
    batch["n_sents"] = len(batch["words"])
    if add_label_wise_instances:
      batch = self._add_label_wise_instances_to_batch(batch)
    batch["words"], batch["seq_len"] = self.pad_sequences(batch["words"])
    if self.config["use_chars"]:
      batch["chars"], _ = self.pad_char_sequences(batch["chars"],
                                                  max_token_length=20)
    if self.config["use_pos_tags"]:
      batch["pos_tags"], _ = self.pad_sequences(batch["pos_tags"])
    batch["batch_size"] = len(batch["words"])
    return batch

  def batchnize_dataset(self, data, batch_size, shuffle=True,
                        add_label_wise_instances=False):
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
        batches.append(self.make_each_batch(batch, add_label_wise_instances))
        batch = defaultdict(list)

      for field in ["sent_id", "words", "chars", "pos_tags",
                    "puncts", "heads", "labels"]:
        batch[field].append(record[field])
      prev_seq_len = seq_len

    if len(batch["words"]) > 0:
      batches.append(self.make_each_batch(batch, add_label_wise_instances))

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


class BERTUnlabeledInstanceBatcher(UnlabeledInstanceBatcher):

  def __init__(self, config):
    super(BERTUnlabeledInstanceBatcher, self).__init__(config)
    self.train_bert_hdf5 = h5py.File(config["train_bert_hdf5"], "r")
    self.valid_bert_hdf5 = h5py.File(config["valid_bert_hdf5"], "r")

  def load_and_add_bert_reps(self, batch, use_train_bert_hdf5):
    if use_train_bert_hdf5:
      batch["bert_rep"] = [self.train_bert_hdf5[str(sent_id)][1:]
                           for sent_id in batch["sent_id"]]
    else:
      n_sents = batch["n_sents"]
      batch_valid = [self.valid_bert_hdf5[str(sent_id)][1:]
                     for sent_id in batch["sent_id"][:n_sents]]
      batch_train = [self.train_bert_hdf5[str(sent_id)][1:]
                     for sent_id in batch["sent_id"][n_sents:]]
      batch["bert_rep"] = batch_valid + batch_train
    return batch

  def _add_random_sents_to_batch(self, batch):
    random.shuffle(self.train_sent_ids)
    for sent_id in self.train_sent_ids[:self.config["k"]]:
      train_sent_record = self.train_sents[sent_id]
      for field in ["sent_id", "words"]:
        batch[field].append(train_sent_record[field])
      if self.config["use_chars"]:
        batch["chars"].append(train_sent_record["chars"])
      if self.config["use_pos_tags"]:
        batch["pos_tags"].append(train_sent_record["pos_tags"])
      batch["neighbor_heads"].append(train_sent_record["heads"])
    return batch

  def _add_neighbor_sents_to_batch(self, batch):
    for sent_id in batch["train_sent_ids"][:self.config["k"]]:
      train_sent_record = self.train_sents[sent_id]
      for field in ["sent_id", "words"]:
        batch[field].append(train_sent_record[field])
      if self.config["use_chars"]:
        batch["chars"].append(train_sent_record["chars"])
      if self.config["use_pos_tags"]:
        batch["pos_tags"].append(train_sent_record["pos_tags"])
      batch["neighbor_heads"].append(train_sent_record["heads"])
    return batch

  def make_each_batch(self, batch, add_neighbor_sents=False):
    batch["n_sents"] = len(batch["words"])

    if "train_sent_ids" in batch:
      batch = self._add_neighbor_sents_to_batch(batch)
    else:
      batch = self._add_random_sents_to_batch(batch)

    batch["words"], batch["seq_len"] = self.pad_sequences(batch["words"])
    if self.config["use_chars"]:
      batch["chars"], _ = self.pad_char_sequences(batch["chars"],
                                                  max_token_length=20)
    if self.config["use_pos_tags"]:
      batch["pos_tags"], _ = self.pad_sequences(batch["pos_tags"])
    batch["neighbor_deps"], batch["neighbor_heads"] = \
      self._flatten(batch["neighbor_heads"], max_len=max(batch["seq_len"]))
    batch["batch_size"] = len(batch["words"])
    return batch

  def batchnize_dataset(self, data, batch_size, shuffle=True,
                        add_neighbor_sents=False):
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


class BERTLabeledInstanceBatcher(LabeledInstanceBatcher):

  def __init__(self, config):
    super(BERTLabeledInstanceBatcher, self).__init__(config)
    self.train_bert_hdf5 = h5py.File(config["train_bert_hdf5"], "r")
    self.valid_bert_hdf5 = h5py.File(config["valid_bert_hdf5"], "r")

  def load_and_add_bert_reps(self, batch, use_train_bert_hdf5):
    if use_train_bert_hdf5:
      batch["bert_rep"] = [self.train_bert_hdf5[str(sent_id)][1:]
                           for sent_id in batch["sent_id"]]
    else:
      batch["bert_rep"] = [self.valid_bert_hdf5[str(sent_id)][1:]
                           for sent_id in batch["sent_id"]]
    return batch

  def make_neighbor_batch_with_all_labels(self, k=1):
    batch = {"sent_id": [], "words": [], "chars": [], "pos_tags": [],
             "class_deps": [], "class_heads": [], "class_labels": []}
    for label, sent_bank in enumerate(self.label_wise_sent_bank):
      sampled_instances = random.sample(sent_bank, k=min(k, len(sent_bank)))
      for sent_id, token_index in sampled_instances:
        train_sent_record = self.train_sents[sent_id]
        for field in ["sent_id", "words", "chars", "pos_tags"]:
          batch[field].append(train_sent_record[field])
        batch["class_deps"].append(token_index)
        batch["class_heads"].append(train_sent_record["heads"][token_index])
        batch["class_labels"].append(train_sent_record["labels"][token_index])

    batch["words"], batch["seq_len"] = self.pad_sequences(batch["words"])
    if self.config["use_chars"]:
      batch["chars"], _ = self.pad_char_sequences(batch["chars"],
                                                  max_token_length=20)
    if self.config["use_pos_tags"]:
      batch["pos_tags"], _ = self.pad_sequences(batch["pos_tags"])
    batch["batch_size"] = len(batch["words"])
    return batch

  def _add_label_wise_instances_to_batch(self, batch):
    for label, sent_bank in enumerate(self.label_wise_sent_bank):
      index = np.random.randint(0, len(sent_bank))
      sent_id, token_index = sent_bank[index]
      train_sent_record = self.train_sents[sent_id]
      for field in ["sent_id", "words", "chars", "pos_tags"]:
        batch[field].append(train_sent_record[field])
      batch["class_deps"].append(token_index)
      batch["class_heads"].append(train_sent_record["heads"][token_index])
    return batch

  def make_each_batch(self, batch, add_label_wise_instances=False):
    batch["n_sents"] = len(batch["words"])
    if add_label_wise_instances:
      batch = self._add_label_wise_instances_to_batch(batch)
    batch["words"], batch["seq_len"] = self.pad_sequences(batch["words"])
    if self.config["use_chars"]:
      batch["chars"], _ = self.pad_char_sequences(batch["chars"],
                                                  max_token_length=20)
    if self.config["use_pos_tags"]:
      batch["pos_tags"], _ = self.pad_sequences(batch["pos_tags"])
    batch["batch_size"] = len(batch["words"])
    return batch

  def batchnize_dataset(self, data, batch_size, shuffle=True,
                        add_label_wise_instances=False):
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
        batches.append(self.make_each_batch(batch, add_label_wise_instances))
        batch = defaultdict(list)

      for field in ["sent_id", "words", "chars", "pos_tags",
                    "puncts", "heads", "labels"]:
        batch[field].append(record[field])
      prev_seq_len = seq_len

    if len(batch["chars"]) > 0:
      batches.append(self.make_each_batch(batch, add_label_wise_instances))

    if shuffle:
      random.shuffle(batches)
    for batch in batches:
      yield batch

