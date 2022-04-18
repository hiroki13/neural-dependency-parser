# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import os
import time

import numpy as np
import tensorflow as tf

from models.unlabeled_models import UnlabeledWeightBasedModel, \
  UnlabeledInstanceBasedModel
from utils.common import load_json, write_json, l2_normalize


VALID_BATCH_SIZE = 128


class LabeledWeightBasedModel(UnlabeledWeightBasedModel):

  def __init__(self, config, batcher, is_train=True):
    super(LabeledWeightBasedModel, self).__init__(config, batcher, is_train)
    if is_train:
      self._build_count_correct_labels_op()

  def _add_placeholders(self):
    if self.cfg["use_bert"]:
      self.bert_rep = tf.placeholder(
        tf.float32, shape=[None, None, self.cfg["bert_dim"]], name="bert")
    if self.cfg["use_words"] or self.cfg["use_bert"] is False:
      self.words = tf.placeholder(
        tf.int32, shape=[None, None], name="words")
    if self.cfg["use_chars"]:
      self.chars = tf.placeholder(
        tf.int32, shape=[None, None, None], name="chars")
    if self.cfg["use_pos_tags"]:
      self.pos_tags = tf.placeholder(
        tf.int32, shape=[None, None], name="pos_tags")

    self.puncts = tf.placeholder(
      tf.int32, shape=[None, None], name="puncts")
    self.seq_len = tf.placeholder(
      tf.int32, shape=[None], name="seq_len")
    self.batch_size = tf.placeholder(
      tf.int32, shape=None, name="batch_size")
    self.heads = tf.placeholder(
      tf.int32, shape=[None, None], name="heads")
    self.labels = tf.placeholder(
      tf.int32, shape=[None, None], name="labels")

    # hyperparameters
    self.is_train = tf.placeholder(tf.bool, name="is_train")
    self.keep_prob = tf.placeholder(tf.float32, name="rnn_keep_probability")
    self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
    if self.cfg["use_bert"]:
      self.bert_drop_rate = tf.placeholder(
        tf.float32, name="bert_dropout_rate")
    self.lr = tf.placeholder(tf.float32, name="learning_rate")

  def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
    feed_dict = {self.puncts: batch["puncts"],
                 self.seq_len: batch["seq_len"],
                 self.batch_size: batch["batch_size"]}

    if self.cfg["use_bert"]:
      feed_dict[self.bert_rep] = batch["bert_rep"]
    if self.cfg["use_words"] or self.cfg["use_bert"] is False:
      feed_dict[self.words] = batch["words"]
    if self.cfg["use_chars"]:
      feed_dict[self.chars] = batch["chars"]
    if self.cfg["use_pos_tags"]:
      feed_dict[self.pos_tags] = batch["pos_tags"]

    if "heads" in batch:
      feed_dict[self.heads] = batch["heads"]
    if "labels" in batch:
      feed_dict[self.labels] = batch["labels"]

    feed_dict[self.keep_prob] = keep_prob
    feed_dict[self.drop_rate] = 1.0 - keep_prob
    if self.cfg["use_bert"]:
      feed_dict[self.bert_drop_rate] = 1.0 - self.cfg["bert_keep_prob"]
    feed_dict[self.is_train] = is_train
    if lr is not None:
      feed_dict[self.lr] = lr

    return feed_dict

  def _build_label_classifier_op(self):
    with tf.variable_scope("label_classifier"):
      self.label_weights = tf.get_variable(name="label_weights",
                                           trainable=True,
                                           shape=[self.label_vocab_size,
                                                  self.cfg["num_units"]])
      self.label_logits = self._compute_logits(self.edge_reps,
                                               self.label_weights)
      print("label classifier: sim={}, scaling_factor={}".format(
        self.cfg["sim"], self.cfg["scaling_factor"]))

  def _build_decoder_op(self):
    self._build_label_classifier_op()

  def _build_loss_op(self):
    self._build_label_loss_op()
    self.loss = self.label_loss
    tf.summary.scalar("loss", self.loss)

  def _build_label_loss_op(self):
    n_tokens = self.seq_len[0]
    batch_index = tf.reshape(tf.range(self.batch_size * (n_tokens-1)),
                             shape=[-1, 1])
    head_index = tf.reshape(self.heads, shape=[-1, 1])
    index = tf.concat([batch_index, head_index], axis=-1)

    logits = tf.reshape(self.label_logits,
                        shape=[self.batch_size * (n_tokens-1),
                               n_tokens,
                               self.label_vocab_size])
    gold_heads = tf.reshape(tf.gather_nd(logits, index),
                            shape=[self.batch_size,
                                   n_tokens-1,
                                   self.label_vocab_size])

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=gold_heads, labels=self.labels)
    self.label_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=-1))
    tf.summary.scalar("label_loss", self.label_loss)

  def _build_predict_op(self):
    self._build_label_predict_op()

  def _build_label_predict_op(self):
    n_tokens = self.seq_len[0]
    batch_index = tf.reshape(tf.range(self.batch_size * (n_tokens-1)),
                             shape=[-1, 1])
    head_index = tf.reshape(self.heads, shape=[-1, 1])
    index = tf.concat([batch_index, head_index], axis=-1)

    logits = tf.reshape(self.label_logits,
                        shape=[self.batch_size * (n_tokens-1),
                               n_tokens,
                               self.label_vocab_size])
    self.logits_of_gold_heads = tf.reshape(tf.gather_nd(logits, index),
                                           shape=[self.batch_size,
                                                  n_tokens - 1,
                                                  self.label_vocab_size])

    self.label_predicts = tf.cast(
      tf.argmax(self.logits_of_gold_heads, axis=-1), tf.int32)

  def _build_label_proba_op(self):
    self.label_proba = tf.nn.softmax(self.logits_of_gold_heads, axis=-1)

  def _build_answer_check_op(self):
    self._build_count_correct_labels_op()

  def _build_count_correct_labels_op(self):
    correct_labels = tf.cast(
      tf.equal(self.label_predicts, self.labels), tf.int32)
    n_tokens = (self.seq_len[0] - 1) * self.batch_size
    if self.cfg["include_puncts"]:
      self.correct_labels = correct_labels
      self.num_tokens = n_tokens
    else:
      not_puncts = 1 - self.puncts
      self.correct_labels = correct_labels * not_puncts
      self.num_tokens = n_tokens - tf.reduce_sum(self.puncts)
    self.num_correct_labels = tf.reduce_sum(self.correct_labels)

  def train_epoch(self, batches):
    loss_total = 0.
    num_tokens = 0
    num_sents = 0
    num_batches = 0
    num_correct_labels = 0
    start_time = time.time()

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=True)
      feed_dict = self._get_feed_dict(batch, is_train=True,
                                      keep_prob=self.cfg["keep_prob"],
                                      lr=self.cfg["lr"])
      outputs = self.sess.run([self.train_op, self.loss,
                               self.num_correct_labels, self.num_tokens],
                              feed_dict)
      _, train_loss, num_cur_correct_labels, num_cur_tokens = outputs

      loss_total += train_loss
      num_correct_labels += num_cur_correct_labels
      num_tokens += num_cur_tokens
      num_sents += batch["batch_size"]

    avg_loss = loss_total / num_batches
    LAS = num_correct_labels / num_tokens
    seconds = time.time() - start_time
    self.logger.info("-- Train set")
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))
    self.logger.info("---- Loss: {:.2f} ({:.2f}/{:d})".format(
      avg_loss, loss_total, num_batches))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, num_correct_labels, num_tokens))
    return avg_loss, loss_total

  def evaluate_epoch(self, batches, data_name):
    num_tokens = 0
    num_sents = 0
    num_batches = 0
    num_correct_labels = 0
    start_time = time.time()

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      num_cur_correct_labels, num_cur_tokens = self.sess.run(
        [self.num_correct_labels, self.num_tokens], feed_dict)

      num_correct_labels += num_cur_correct_labels
      num_tokens += num_cur_tokens
      num_sents += batch["batch_size"]

    LAS = num_correct_labels / num_tokens
    seconds = time.time() - start_time
    self.logger.info("-- {} set".format(data_name))
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, num_correct_labels, num_tokens))
    return LAS

  def train(self):
    self.logger.info(str(self.cfg))
    write_json(os.path.join(self.cfg["checkpoint_path"], "config.json"),
               self.cfg)

    epochs = self.cfg["epochs"]
    train_path = self.cfg["train_set"]
    valid_path = self.cfg["valid_set"]
    valid_batch_size = max(self.cfg["batch_size"], VALID_BATCH_SIZE)
    valid_set = list(self.batcher.batchnize_dataset(
      valid_path, valid_batch_size, shuffle=True))
    best_LAS = -np.inf
    init_lr = self.cfg["lr"]

    self.log_trainable_variables()
    self.logger.info("Start training...")
    self._add_summary()
    for epoch in range(1, epochs + 1):
      self.logger.info('Epoch {}/{}:'.format(epoch, epochs))

      train_set = self.batcher.batchnize_dataset(
        train_path, self.cfg["batch_size"], shuffle=True)
      _ = self.train_epoch(train_set)

      if self.cfg["use_lr_decay"]:  # learning rate decay
        self.cfg["lr"] = max(init_lr / (1.0 + self.cfg["lr_decay"] * epoch),
                             self.cfg["minimal_lr"])

      cur_valid_LAS = self.evaluate_epoch(valid_set, "Valid")

      if cur_valid_LAS > best_LAS:
        best_LAS = cur_valid_LAS
        self.save_session(epoch)
        self.logger.info(
          "-- new BEST LAS on Valid set: {:>7.2%}".format(best_LAS))

    self.train_writer.close()
    self.test_writer.close()

  def eval(self, preprocessor):
    self.logger.info(str(self.cfg))
    raw_data = load_json(self.cfg["data_path"])[:self.cfg["data_size"]]
    _, indexed_data = self._preprocess_input_data(preprocessor)

    #############
    # Main loop #
    #############
    num_tokens = 0
    num_sents = len(indexed_data)
    num_correct_labels = 0
    start_time = time.time()

    print("PREDICTION START")
    for sent_id, (record, indexed_sent) in enumerate(zip(raw_data,
                                                         indexed_data)):
      assert sent_id == record["sent_id"] == indexed_sent["sent_id"]
      if (sent_id + 1) % 100 == 0:
        print("%d" % (sent_id + 1), flush=True, end=" ")

      batch = self.make_one_batch(indexed_sent)
      if self.cfg["use_gold_heads"] or "predicted_heads" not in record:
        batch["heads"] = [record["heads"]]
      else:
        batch["heads"] = [record["predicted_heads"]]

      ##############
      # Prediction #
      ##############
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      predicted_labels = self.sess.run([self.label_predicts], feed_dict)[0][0]

      ###############
      # Add results #
      ###############
      record["predicted_labels"] = [self.rev_label_dict[label]
                                    for label in predicted_labels]

      #########
      # Count #
      #########
      puncts = None if self.cfg["include_puncts"] else indexed_sent["puncts"]
      num_cur_correct_labels = self._count_correct_labels(
        gold_heads=record["heads"],
        predicted_heads=batch["heads"][0],
        gold_labels=record["labels"],
        predicted_labels=record["predicted_labels"],
        puncts=puncts
      )
      num_correct_labels += num_cur_correct_labels
      # -1 means excluding the ROOT node
      num_tokens += batch["seq_len"][0] - 1
      # - np.sum(batch["puncts"]) means excluding punctuations
      if self.cfg["include_puncts"] is False:
        num_tokens -= np.sum(batch["puncts"])

    LAS = num_correct_labels / num_tokens
    seconds = time.time() - start_time
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))
    if self.cfg["use_gold_heads"] or "predicted_heads" not in raw_data[0]:
      self.logger.info("---- LAS (gold heads):{:>7.2%} ({:>6}/{:>6})".format(
        LAS, num_correct_labels, num_tokens))
    else:
      self.logger.info("---- LAS (pred heads):{:>7.2%} ({:>6}/{:>6})".format(
        LAS, num_correct_labels, num_tokens))

    data_name = os.path.splitext(os.path.basename(self.cfg["data_path"]))[0]
    if self.cfg["output_file"]:
      if self.cfg["output_file"].endswith(".json"):
        file_name = self.cfg["output_file"]
      else:
        file_name = self.cfg["output_file"] + ".json"
      data_path = os.path.join(self.cfg["checkpoint_path"], file_name)
    else:
      data_path = os.path.join(self.cfg["checkpoint_path"],
                               data_name + ".predicted_labels.json")
    write_json(data_path, raw_data)

  def eval_and_proba(self, preprocessor):
    self.logger.info(str(self.cfg))
    raw_data = load_json(self.cfg["data_path"])[:self.cfg["data_size"]]
    _, indexed_data = self._preprocess_input_data(preprocessor)

    #############
    # Main loop #
    #############
    num_tokens = 0
    num_sents = len(indexed_data)
    num_correct_labels = 0
    start_time = time.time()

    print("PREDICTION START")
    for sent_id, (record, indexed_sent) in enumerate(zip(raw_data,
                                                         indexed_data)):
      assert sent_id == record["sent_id"] == indexed_sent["sent_id"]
      if (sent_id + 1) % 100 == 0:
        print("%d" % (sent_id + 1), flush=True, end=" ")

      batch = self.make_one_batch(indexed_sent)
      if self.cfg["use_gold_heads"] or "predicted_heads" not in record:
        batch["heads"] = [record["heads"]]
      else:
        batch["heads"] = [record["predicted_heads"]]

      ##############
      # Prediction #
      ##############
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      outputs = self.sess.run([self.label_predicts, self.label_proba],
                              feed_dict)
      predicted_labels = outputs[0][0]
      label_proba = outputs[1][0]

      ###############
      # Add results #
      ###############
      record["predicted_labels"] = [self.rev_label_dict[label]
                                    for label in predicted_labels]
      record["label_proba"] = [[float(p) for p in ps] for ps in label_proba]

      #########
      # Count #
      #########
      puncts = None if self.cfg["include_puncts"] else indexed_sent["puncts"]
      num_cur_correct_labels = self._count_correct_labels(
        gold_heads=record["heads"],
        predicted_heads=batch["heads"][0],
        gold_labels=record["labels"],
        predicted_labels=record["predicted_labels"],
        puncts=puncts
      )
      num_correct_labels += num_cur_correct_labels
      # -1 means excluding the ROOT node
      num_tokens += batch["seq_len"][0] - 1
      # - np.sum(batch["puncts"]) means excluding punctuations
      if self.cfg["include_puncts"] is False:
        num_tokens -= np.sum(batch["puncts"])

    LAS = num_correct_labels / num_tokens
    seconds = time.time() - start_time
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))
    if self.cfg["use_gold_heads"] or "predicted_heads" not in raw_data[0]:
      self.logger.info("---- LAS (gold heads):{:>7.2%} ({:>6}/{:>6})".format(
        LAS, num_correct_labels, num_tokens))
    else:
      self.logger.info("---- LAS (pred heads):{:>7.2%} ({:>6}/{:>6})".format(
        LAS, num_correct_labels, num_tokens))

    data_name = os.path.splitext(os.path.basename(self.cfg["data_path"]))[0]
    if self.cfg["output_file"]:
      if self.cfg["output_file"].endswith(".json"):
        file_name = self.cfg["output_file"]
      else:
        file_name = self.cfg["output_file"] + ".json"
      data_path = os.path.join(self.cfg["checkpoint_path"], file_name)
    else:
      data_path = os.path.join(self.cfg["checkpoint_path"],
                               data_name + ".predicted_labels.json")
    write_json(data_path, raw_data)


class LabeledInstanceBasedModel(UnlabeledInstanceBasedModel):

  def _add_placeholders(self):
    if self.cfg["use_bert"]:
      self.bert_rep = tf.placeholder(
        tf.float32, shape=[None, None, self.cfg["bert_dim"]], name="bert")
    if self.cfg["use_words"] or self.cfg["use_bert"] is False:
      self.words = tf.placeholder(
        tf.int32, shape=[None, None], name="words")
    if self.cfg["use_chars"]:
      self.chars = tf.placeholder(
        tf.int32, shape=[None, None, None], name="chars")
    if self.cfg["use_pos_tags"]:
      self.pos_tags = tf.placeholder(
        tf.int32, shape=[None, None], name="pos_tags")

    self.puncts = tf.placeholder(
      tf.int32, shape=[None, None], name="puncts")
    self.seq_len = tf.placeholder(
      tf.int32, shape=[None], name="seq_len")
    self.batch_size = tf.placeholder(
      tf.int32, shape=None, name="batch_size")
    self.n_sents = tf.placeholder(
      tf.int32, shape=None, name="n_sents")

    self.class_deps = tf.placeholder(
      tf.int32, shape=None, name="class_dep_indices")
    self.class_heads = tf.placeholder(
      tf.int32, shape=None, name="class_head_indices")
    self.neighbor_reps = tf.placeholder(
      tf.float32, shape=[self.label_vocab_size, self.cfg["num_units"]],
      name="neighbor_reps")

    self.heads = tf.placeholder(
      tf.int32, shape=[None, None], name="heads")
    self.labels = tf.placeholder(
      tf.int32, shape=[None, None], name="labels")
    self.n_each_label = tf.placeholder(
      tf.float32, shape=[None], name="n_each_label")

    # hyperparameters
    self.is_train = tf.placeholder(tf.bool, name="is_train")
    self.keep_prob = tf.placeholder(tf.float32, name="rnn_keep_probability")
    self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
    if self.cfg["use_bert"]:
      self.bert_drop_rate = tf.placeholder(
        tf.float32, name="bert_dropout_rate")
    self.lr = tf.placeholder(tf.float32, name="learning_rate")

  def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
    feed_dict = {self.puncts: batch["puncts"],
                 self.seq_len: batch["seq_len"],
                 self.batch_size: batch["batch_size"],
                 self.n_sents: batch["n_sents"]}
    if self.cfg["use_bert"]:
      feed_dict[self.bert_rep] = batch["bert_rep"]
    if self.cfg["use_words"] or self.cfg["use_bert"] is False:
      feed_dict[self.words] = batch["words"]
    if self.cfg["use_chars"]:
      feed_dict[self.chars] = batch["chars"]
    if self.cfg["use_pos_tags"]:
      feed_dict[self.pos_tags] = batch["pos_tags"]

    if "class_deps" in batch:
      feed_dict[self.class_deps] = batch["class_deps"]
    if "class_heads" in batch:
      feed_dict[self.class_heads] = batch["class_heads"]
    if "neighbor_reps" in batch:
      feed_dict[self.neighbor_reps] = batch["neighbor_reps"]

    if "heads" in batch:
      feed_dict[self.heads] = batch["heads"]
    if "labels" in batch:
      feed_dict[self.labels] = batch["labels"]

    if is_train:
      feed_dict[self.n_each_label] = [1 for _ in range(self.label_vocab_size)]
    elif self.use_all_instances:
      feed_dict[self.n_each_label] = self.batcher.n_each_label
    else:
      feed_dict[self.n_each_label] = [
        min(self.cfg["k"], n_labels) for n_labels in self.batcher.n_each_label]

    feed_dict[self.keep_prob] = keep_prob
    feed_dict[self.drop_rate] = 1.0 - keep_prob
    if self.cfg["use_bert"]:
      feed_dict[self.bert_drop_rate] = 1.0 - self.cfg["bert_keep_prob"]
    feed_dict[self.is_train] = is_train
    if lr is not None:
      feed_dict[self.lr] = lr
    return feed_dict

  def _build_encoder_op(self):
    self._build_rnn_op()
    self._build_head_and_dep_projection_op()
    self._build_edge_dense_layer_op()
    self._build_anchor_dep_and_head_rep_op()
    self._build_anchor_edge_rep_op()
    self._build_class_dep_and_head_rep_op()
    self._build_class_edge_rep_op()

  def _build_anchor_dep_and_head_rep_op(self):
    with tf.variable_scope("anchor_dep_and_head_rep"):
      n_tokens = self.seq_len[0]
      n_sents = self.n_sents

      # 1D: n_sens, 2D: n_tokens, 3D: dim
      head_without_pads = self.head_rep[:n_sents, :n_tokens]
      # 1D: n_sents * n_tokens, 2D: dim
      head_rep = tf.reshape(head_without_pads,
                            shape=[n_sents * n_tokens,
                                   self.cfg["num_units"]])

      # 1D: n_sents, 2D: 1
      index = tf.expand_dims(tf.range(n_sents) * n_tokens, axis=1)
      # 1D: n_sents * (n_tokens-1)
      head_indices = tf.reshape(self.heads + index, shape=[-1])
      # 1D: n_sents, 2D: n_tokens-1, 3D: dim
      self.anchor_head_rep = tf.reshape(tf.gather(head_rep, head_indices),
                                        shape=[n_sents,
                                               n_tokens-1,
                                               self.cfg["num_units"]])
      # 1D: n_sents, 2D: n_tokens-1, 3D: dim
      self.anchor_dep_rep = self.dep_rep[:n_sents, 1:n_tokens]

  def _build_anchor_edge_rep_op(self):
    with tf.variable_scope("anchor_edge_rep"):
      # 1D: n_sents, 2D: n_tokens-1, 3D: dim
      edge_rep = self._create_edge_rep(self.anchor_dep_rep,
                                       self.anchor_head_rep)
      self.anchor_edge_reps = tf.tensordot(edge_rep, self.edge_rep_dense,
                                           axes=[-1, -1])
      print("anchor edge rep shape: {}".format(
        self.anchor_edge_reps.get_shape().as_list()))

  def _build_class_dep_and_head_rep_op(self):
    with tf.variable_scope("class_dep_and_head_rep"):
      dep_rep = self.dep_rep[self.n_sents:, 1:]
      head_rep = self.head_rep[self.n_sents:]

      class_index = tf.reshape(tf.range(self.label_vocab_size), shape=[-1, 1])
      dep_index = tf.reshape(self.class_deps, shape=[-1, 1])
      dep_index = tf.concat([class_index, dep_index], axis=-1)
      head_index = tf.reshape(self.class_heads, shape=[-1, 1])
      head_index = tf.concat([class_index, head_index], axis=-1)

      # 1D: n_classes, 2D: dim
      self.class_dep_rep = tf.gather_nd(dep_rep, dep_index)
      self.class_head_rep = tf.gather_nd(head_rep, head_index)

  def _build_class_edge_rep_op(self):
    with tf.variable_scope("class_edge_rep"):
      # 1D: n_classes, 2D: dim
      edge_rep = self._create_edge_rep(self.class_dep_rep,
                                       self.class_head_rep)
      self.class_edge_reps = tf.tensordot(edge_rep, self.edge_rep_dense,
                                          axes=[-1, -1])
      print("class edge rep shape: {}".format(
        self.class_edge_reps.get_shape().as_list()))

  def _build_decoder_op(self):
    self._build_train_logits_op()
    self._build_logits_op()

  def _build_train_logits_op(self):
    with tf.name_scope("train_logits"):
      # 1D: n_sents, 2D: n_tokens-1, 3D: n_classes
      self.train_logits = self._compute_logits(self.anchor_edge_reps,
                                               self.class_edge_reps)

  def _build_logits_op(self):
    with tf.name_scope("logits"):
      if self.cfg["sim"] == "dot":
        anchor_edge_reps = self.anchor_edge_reps
      else:
        anchor_edge_reps = tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1)
      # 1D: n_sents, 2D: n_tokens-1, 3D: n_classes
      # neighbor_reps have already normalized when self.cfg["sim"] == "cos"
      logits = tf.tensordot(anchor_edge_reps, self.neighbor_reps,
                            axes=[-1, -1])
      self.logits = logits / self.n_each_label

  def _build_loss_op(self):
    self._build_label_loss_op()
    self.loss = self.label_loss
    tf.summary.scalar("loss", self.loss)

  def _build_label_loss_op(self):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.train_logits, labels=self.labels)
    self.label_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=-1))

  def _build_label_proba_op(self):
    self.label_proba = tf.nn.softmax(self.logits, axis=-1)

  def _build_predict_op(self):
    self._build_train_label_prediction_op()
    self._build_label_prediction_op()

  def _build_train_label_prediction_op(self):
    self.train_label_predicts = tf.cast(
      tf.argmax(self.train_logits, axis=-1), tf.int32)

  def _build_label_prediction_op(self):
    # 1D: n_sents, 2D: n_tokens-1, 3D: n_classes
    self.label_predicts = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

  def _build_answer_check_op(self):
    self._build_count_correct_train_labels_op()
    self._build_count_correct_labels_op()

  def _build_count_correct_train_labels_op(self):
    correct_labels = tf.cast(
      tf.equal(self.train_label_predicts, self.labels), tf.int32)
    n_tokens = (self.seq_len[0] - 1) * self.n_sents
    if self.cfg["include_puncts"]:
      correct_labels = correct_labels
      self.num_train_tokens = n_tokens
    else:
      not_puncts = 1 - self.puncts
      correct_labels = correct_labels * not_puncts
      self.num_train_tokens = n_tokens - tf.reduce_sum(self.puncts)
    self.num_correct_train_labels = tf.reduce_sum(correct_labels)

  def _build_count_correct_labels_op(self):
    correct_labels = tf.cast(
      tf.equal(self.label_predicts, self.labels), tf.int32)
    n_tokens = (self.seq_len[0] - 1) * self.n_sents
    if self.cfg["include_puncts"]:
      self.correct_labels = correct_labels
      self.num_tokens = n_tokens
    else:
      not_puncts = 1 - self.puncts
      self.correct_labels = correct_labels * not_puncts
      self.num_tokens = n_tokens - tf.reduce_sum(self.puncts)
    self.num_correct_labels = tf.reduce_sum(self.correct_labels)

  def _add_bert_reps(self, batch, use_train_bert_hdf5):
    batch = self.batcher.load_and_add_bert_reps(batch, use_train_bert_hdf5)
    return self.batcher.pad_bert_reps(batch)

  def precompute_edge_rep(self, dim_rep):
    self.logger.info("------ Precomputing train data")

    batch_size = max(self.cfg["batch_size"], VALID_BATCH_SIZE)
    batches = self.batcher.batchnize_dataset(self.cfg["train_set"],
                                             batch_size,
                                             shuffle=True,
                                             add_label_wise_instances=False)
    start_time = time.time()
    num_sents = 0
    precomp_edge_reps = np.zeros(shape=(self.label_vocab_size, dim_rep))
    for batch in batches:
      num_sents += batch["n_sents"]

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=True)
      feed_dict = self._get_feed_dict(batch)
      edge_reps = self.sess.run([self.anchor_edge_reps], feed_dict)[0]

      for sent_labels, sent_edge_reps in zip(batch["labels"], edge_reps):
        for r, rep in zip(sent_labels, sent_edge_reps):
          if self.sim == "cos":
            rep = l2_normalize(rep)
          precomp_edge_reps[r] += rep
    self.precomputed_edge_rep = precomp_edge_reps
    seconds = time.time() - start_time
    self.logger.info("------ Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))

  def precompute_edge_rep_for_random_sampling(self):
    self.logger.info("------ Precomputing train data")

    f_hdf5_path = os.path.join(self.cfg["checkpoint_path"],
                               "train.label_wise_edge_reps.hdf5")
    if self.batcher.label_wise_edge_reps_hdf5 is not None:
      self.batcher.label_wise_edge_reps_hdf5.close()
    f_hdf5 = h5py.File(f_hdf5_path, 'w')

    batch_size = max(self.cfg["batch_size"], VALID_BATCH_SIZE)
    batches = self.batcher.batchnize_dataset(self.cfg["train_set"],
                                             batch_size,
                                             shuffle=True,
                                             add_label_wise_instances=False)
    start_time = time.time()
    num_sents = 0
    for batch in batches:
      num_sents += batch["n_sents"]

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=True)
      feed_dict = self._get_feed_dict(batch)
      edge_reps = self.sess.run([self.anchor_edge_reps], feed_dict)[0]

      for sent_id, sent_edge_reps in zip(batch["sent_id"], edge_reps):
        f_hdf5.create_dataset(name='{}'.format(sent_id),
                              dtype='float32',
                              data=sent_edge_reps)
    f_hdf5.close()
    self.batcher.label_wise_edge_reps_hdf5 = h5py.File(f_hdf5_path, 'r')

    seconds = time.time() - start_time
    self.logger.info("------ Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))

  def add_precomputed_rep_to_batch(self, batch):
    if self.use_all_instances:
      batch["neighbor_reps"] = self.precomputed_edge_rep
    else:
      reps = self.batcher.get_precomputed_label_wise_reps(k=self.k)
      batch["neighbor_reps"] = reps
    return batch

  def train_epoch(self, batches):
    loss_total = 0.
    num_tokens = 0
    num_sents = 0
    num_batches = 0
    num_correct_labels = 0
    start_time = time.time()

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=True)
      feed_dict = self._get_feed_dict(batch,
                                      is_train=True,
                                      keep_prob=self.cfg["keep_prob"],
                                      lr=self.cfg["lr"])
      outputs = self.sess.run([self.train_op,
                               self.loss,
                               self.num_correct_train_labels,
                               self.num_train_tokens],
                              feed_dict)
      _, train_loss, num_cur_correct_labels, num_cur_tokens = outputs

      loss_total += train_loss
      num_correct_labels += num_cur_correct_labels
      num_tokens += num_cur_tokens
      num_sents += batch["n_sents"]

    avg_loss = loss_total / num_batches
    LAS = num_correct_labels / num_tokens
    seconds = time.time() - start_time
    self.logger.info("-- Train set")
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))
    self.logger.info("---- Loss: {:.2f} ({:.2f}/{:d})".format(
      avg_loss, loss_total, num_batches))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, num_correct_labels, num_tokens))
    return avg_loss, loss_total

  def evaluate_epoch(self, batches, data_name):
    if self.use_all_instances:
      self.precompute_edge_rep(dim_rep=self.cfg["num_units"])
    else:
      self.precompute_edge_rep_for_random_sampling()

    num_tokens = 0
    num_sents = 0
    num_batches = 0
    num_correct_labels = 0
    start_time = time.time()

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      batch = self.add_precomputed_rep_to_batch(batch)
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      num_cur_correct_labels, num_cur_tokens = self.sess.run(
        [self.num_correct_labels, self.num_tokens], feed_dict)

      num_correct_labels += num_cur_correct_labels
      num_tokens += num_cur_tokens
      num_sents += batch["n_sents"]

    LAS = num_correct_labels / num_tokens
    seconds = time.time() - start_time
    self.logger.info("-- {} set".format(data_name))
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, num_correct_labels, num_tokens))
    return LAS

  def train(self):
    self.logger.info(str(self.cfg))
    write_json(os.path.join(self.cfg["checkpoint_path"], "config.json"),
               self.cfg)

    epochs = self.cfg["epochs"]
    train_path = self.cfg["train_set"]
    valid_path = self.cfg["valid_set"]
    valid_batch_size = max(self.cfg["batch_size"], VALID_BATCH_SIZE)
    valid_set = list(self.batcher.batchnize_dataset(
      valid_path, valid_batch_size,
      shuffle=True, add_label_wise_instances=False))
    best_LAS = -np.inf
    init_lr = self.cfg["lr"]

    self.log_trainable_variables()
    self.logger.info("Start training...")
    self._add_summary()
    for epoch in range(1, epochs + 1):
      self.logger.info('Epoch {}/{}:'.format(epoch, epochs))

      train_set = self.batcher.batchnize_dataset(
        train_path, self.cfg["batch_size"],
        shuffle=True, add_label_wise_instances=True)
      _ = self.train_epoch(train_set)

      if self.cfg["use_lr_decay"]:  # learning rate decay
        self.cfg["lr"] = max(init_lr / (1.0 + self.cfg["lr_decay"] * epoch),
                             self.cfg["minimal_lr"])

      cur_valid_LAS = self.evaluate_epoch(valid_set, "Valid")

      if cur_valid_LAS > best_LAS:
        best_LAS = cur_valid_LAS
        self.save_session(epoch)
        self.logger.info(
          "-- new BEST LAS on Valid set: {:>7.2%}".format(best_LAS))

    self.train_writer.close()
    self.test_writer.close()

  def eval(self, preprocessor):
    self.logger.info(str(self.cfg))
    raw_data = load_json(self.cfg["data_path"])[:self.cfg["data_size"]]
    _, indexed_data = self._preprocess_input_data(preprocessor)

    ################################################
    # Precomputing edge reps of training instances #
    ################################################
    if self.use_all_instances:
      self.precompute_edge_rep(dim_rep=self.cfg["num_units"])
    else:
      self.precompute_edge_rep_for_random_sampling()

    #############
    # Main loop #
    #############
    num_tokens = 0
    num_sents = len(indexed_data)
    num_correct_labels = 0
    start_time = time.time()

    print("PREDICTION START")
    for sent_id, (record, indexed_sent) in enumerate(zip(raw_data,
                                                         indexed_data)):
      assert sent_id == record["sent_id"] == indexed_sent["sent_id"]
      if (sent_id + 1) % 100 == 0:
        print("%d" % (sent_id + 1), flush=True, end=" ")

      batch = self.make_one_batch(indexed_sent)
      batch = self.add_precomputed_rep_to_batch(batch)
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      if self.cfg["use_gold_heads"] or "predicted_heads" not in record:
        batch["heads"] = [record["heads"]]
      else:
        batch["heads"] = [record["predicted_heads"]]

      ##############
      # Prediction #
      ##############
      feed_dict = self._get_feed_dict(batch)
      predicted_labels = self.sess.run([self.label_predicts], feed_dict)[0][0]

      ###############
      # Add results #
      ###############
      record["predicted_labels"] = [self.rev_label_dict[label]
                                    for label in predicted_labels]

      #########
      # Count #
      #########
      puncts = None if self.cfg["include_puncts"] else indexed_sent["puncts"]
      num_cur_correct_labels = self._count_correct_labels(
        gold_heads=record["heads"],
        predicted_heads=batch["heads"][0],
        gold_labels=record["labels"],
        predicted_labels=record["predicted_labels"],
        puncts=puncts
      )
      num_correct_labels += num_cur_correct_labels
      # -1 means excluding the ROOT node
      num_tokens += batch["seq_len"][0] - 1
      # - np.sum(batch["puncts"]) means excluding punctuations
      if self.cfg["include_puncts"] is False:
        num_tokens -= np.sum(batch["puncts"])

    #################
    # Print results #
    #################
    LAS = num_correct_labels / num_tokens
    seconds = time.time() - start_time
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))
    if self.cfg["use_gold_heads"] or "predicted_heads" not in raw_data[0]:
      self.logger.info("---- LAS (gold heads):{:>7.2%} ({:>6}/{:>6})".format(
        LAS, num_correct_labels, num_tokens))
    else:
      self.logger.info("---- LAS (pred heads):{:>7.2%} ({:>6}/{:>6})".format(
        LAS, num_correct_labels, num_tokens))

    ################
    # Save results #
    ################
    data_name = os.path.splitext(os.path.basename(self.cfg["data_path"]))[0]
    if self.cfg["output_file"]:
      if self.cfg["output_file"].endswith(".json"):
        file_name = self.cfg["output_file"]
      else:
        file_name = self.cfg["output_file"] + ".json"
      data_path = os.path.join(self.cfg["checkpoint_path"], file_name)
    else:
      data_path = os.path.join(self.cfg["checkpoint_path"],
                               data_name + ".predicted_labels.json")
    write_json(data_path, raw_data)

  def eval_and_proba(self, preprocessor):
    self.logger.info(str(self.cfg))
    raw_data = load_json(self.cfg["data_path"])[:self.cfg["data_size"]]
    _, indexed_data = self._preprocess_input_data(preprocessor)
    self.precompute_edge_rep(dim_rep=self.cfg["num_units"])

    #############
    # Main loop #
    #############
    num_tokens = 0
    num_sents = len(indexed_data)
    num_correct_labels = 0
    start_time = time.time()

    print("PREDICTION START")
    for sent_id, (record, indexed_sent) in enumerate(zip(raw_data,
                                                         indexed_data)):
      assert sent_id == record["sent_id"] == indexed_sent["sent_id"]
      if (sent_id + 1) % 100 == 0:
        print("%d" % (sent_id + 1), flush=True, end=" ")

      batch = self.make_one_batch(indexed_sent)
      batch = self.add_precomputed_rep_to_batch(batch)
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      if self.cfg["use_gold_heads"] or "predicted_heads" not in record:
        batch["heads"] = [record["heads"]]
      else:
        batch["heads"] = [record["predicted_heads"]]

      ##############
      # Prediction #
      ##############
      feed_dict = self._get_feed_dict(batch)
      outputs = self.sess.run([self.label_predicts, self.label_proba],
                              feed_dict)
      predicted_labels = outputs[0][0]
      label_proba = outputs[1][0]

      ###############
      # Add results #
      ###############
      record["predicted_labels"] = [self.rev_label_dict[label]
                                    for label in predicted_labels]
      record["label_proba"] = [[float(p) for p in ps] for ps in label_proba]
      #########
      # Count #
      #########
      puncts = None if self.cfg["include_puncts"] else indexed_sent["puncts"]
      num_cur_correct_labels = self._count_correct_labels(
        gold_heads=record["heads"],
        predicted_heads=batch["heads"][0],
        gold_labels=record["labels"],
        predicted_labels=record["predicted_labels"],
        puncts=puncts
      )
      num_correct_labels += num_cur_correct_labels
      # -1 means excluding the ROOT node
      num_tokens += batch["seq_len"][0] - 1
      # - np.sum(batch["puncts"]) means excluding punctuations
      if self.cfg["include_puncts"] is False:
        num_tokens -= np.sum(batch["puncts"])

    #################
    # Print results #
    #################
    LAS = num_correct_labels / num_tokens
    seconds = time.time() - start_time
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))
    if self.cfg["use_gold_heads"] or "predicted_heads" not in raw_data[0]:
      self.logger.info("---- LAS (gold heads):{:>7.2%} ({:>6}/{:>6})".format(
        LAS, num_correct_labels, num_tokens))
    else:
      self.logger.info("---- LAS (pred heads):{:>7.2%} ({:>6}/{:>6})".format(
        LAS, num_correct_labels, num_tokens))

    ################
    # Save results #
    ################
    data_name = os.path.splitext(os.path.basename(self.cfg["data_path"]))[0]
    if self.cfg["output_file"]:
      if self.cfg["output_file"].endswith(".json"):
        file_name = self.cfg["output_file"]
      else:
        file_name = self.cfg["output_file"] + ".json"
      data_path = os.path.join(self.cfg["checkpoint_path"], file_name)
    else:
      data_path = os.path.join(self.cfg["checkpoint_path"],
                               data_name + ".predicted_labels.json")
    write_json(data_path, raw_data)


class CosFaceLabeler(LabeledInstanceBasedModel):

  def _build_label_loss_op(self):
    margin = 0.35
    mask = tf.one_hot(tf.cast(self.labels, tf.int32),
                      depth=self.label_vocab_size,
                      name='one_hot_mask')
    logits_with_margin = tf.subtract(self.train_logits, margin)
    logits = tf.where(mask == 1., logits_with_margin, self.train_logits)
    logits = tf.multiply(logits, 64.)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.labels)
    self.label_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=-1))
