# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time

import numpy as np
import tensorflow as tf

from models.labeled_models import LabeledWeightBasedModel, \
  LabeledInstanceBasedModel
from utils.common import load_json, write_json, l2_normalize


VALID_BATCH_SIZE = 128


class MultiTaskWeightBasedModel(LabeledWeightBasedModel):

  def _build_edge_rep_op(self):
    with tf.variable_scope("edge_rep"):
      self.u_edge_rep_dense = tf.get_variable(
        name="u_edge_rep_dense", shape=[self.cfg["num_units"],
                                        self.cfg["num_units"]])
      self.l_edge_rep_dense = tf.get_variable(
        name="l_edge_rep_dense", shape=[self.cfg["num_units"],
                                        self.cfg["num_units"]])

      n_tokens = self.seq_len[0]
      n_sents = self.batch_size

      dep_rep = tf.expand_dims(
        self.dep_rep[:n_sents, 1:n_tokens], axis=2)
      head_rep = tf.expand_dims(
        self.head_rep[:n_sents, :n_tokens], axis=1)

      u_edge_rep = tf.multiply(dep_rep, head_rep)
      l_edge_rep = tf.subtract(dep_rep, head_rep)

      # 1D: batch_size, 2D: n_tokens (dep), 3D: n_tokens (head), 4D: dim
      u_edge_rep = tf.tensordot(
        u_edge_rep, self.u_edge_rep_dense, axes=[-1, -1])
      l_edge_rep = tf.tensordot(
        l_edge_rep, self.l_edge_rep_dense, axes=[-1, -1])
      self.edge_reps = tf.concat([u_edge_rep, l_edge_rep], axis=-1)
      print("edge rep shape: {}".format(
        self.edge_reps.get_shape().as_list()))

  def _build_head_classifier_op(self):
    with tf.variable_scope("head_classifier"):
      self.head_weights = tf.get_variable(name="head_weights",
                                          trainable=True,
                                          shape=self.cfg["num_units"])
      self.head_logits = self._compute_logits(
        self.edge_reps[:, :, :, :self.cfg["num_units"]], self.head_weights)
      print("head classifier: sim={}, scaling_factor={}".format(
        self.cfg["sim"], self.cfg["scaling_factor"]))

  def _build_label_classifier_op(self):
    with tf.variable_scope("label_classifier"):
      self.label_weights = tf.get_variable(name="label_weights",
                                           trainable=True,
                                           shape=[self.label_vocab_size,
                                                  self.cfg["num_units"]])
      self.label_logits = self._compute_logits(
        self.edge_reps[:, :, :, self.cfg["num_units"]:], self.label_weights)
      print("label classifier: sim={}, scaling_factor={}".format(
        self.cfg["sim"], self.cfg["scaling_factor"]))

  def _build_label_predict_op(self):
    n_tokens = self.seq_len[0]
    batch_index = tf.reshape(tf.range(self.batch_size * (n_tokens - 1)),
                             shape=[-1, 1])
    head_index = tf.reshape(self.head_predicts, shape=[-1, 1])
    index = tf.concat([batch_index, head_index], axis=-1)

    logits = tf.reshape(self.label_logits,
                        shape=[self.batch_size * (n_tokens - 1),
                               n_tokens,
                               self.label_vocab_size])
    logits_of_pred_heads = tf.reshape(tf.gather_nd(logits, index),
                                      shape=[self.batch_size,
                                             n_tokens - 1,
                                             self.label_vocab_size])
    self.label_predicts = tf.cast(
      tf.argmax(logits_of_pred_heads, axis=-1), tf.int32)

  def _build_count_correct_labels_op(self):
    correct_labels = tf.cast(
      tf.equal(self.label_predicts, self.labels), tf.int32)
    self.correct_labels = correct_labels * self.correct_heads
    self.num_correct_labels = tf.reduce_sum(self.correct_labels)

  def _build_decoder_op(self):
    self._build_head_classifier_op()
    self._build_label_classifier_op()

  def _build_loss_op(self):
    self._build_head_loss_op()
    self._build_label_loss_op()
    self.loss = self.head_loss + self.label_loss
    tf.summary.scalar("loss", self.loss)

  def _build_predict_op(self):
    self._build_head_prediction_op()
    self._build_label_predict_op()

  def _build_answer_check_op(self):
    self._build_count_correct_heads_op()
    self._build_count_correct_labels_op()

  def train_epoch(self, batches):
    loss_total = 0.
    n_tokens = 0
    n_sents = 0
    n_batches = 0
    n_correct_heads = 0
    n_correct_labels = 0
    start_time = time.time()

    for batch in batches:
      n_batches += 1
      if n_batches % 100 == 0:
        print("%d" % n_batches, flush=True, end=" ")

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=True)
      feed_dict = self._get_feed_dict(
        batch, is_train=True, keep_prob=self.cfg["keep_prob"],
        lr=self.cfg["lr"])
      outputs = self.sess.run(
        [self.train_op, self.loss, self.num_correct_heads,
         self.num_correct_labels, self.num_tokens], feed_dict)
      loss_total += outputs[1]
      n_correct_heads += outputs[2]
      n_correct_labels += outputs[3]
      n_tokens += outputs[4]
      n_sents += batch["batch_size"]

    avg_loss = loss_total / n_batches
    UAS = n_correct_heads / n_tokens
    LAS = n_correct_labels / n_tokens
    seconds = time.time() - start_time
    self.logger.info("-- Train set")
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, n_sents / seconds))
    self.logger.info("---- Loss: {:.2f} ({:.2f}/{:d})".format(
      avg_loss, loss_total, n_batches))
    self.logger.info("---- UAS:{:>7.2%} ({:>6}/{:>6})".format(
      UAS, n_correct_heads, n_tokens))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, n_correct_labels, n_tokens))
    return avg_loss, loss_total

  def evaluate_epoch(self, batches, data_name):
    n_tokens = 0
    n_sents = 0
    n_batches = 0
    n_correct_heads = 0
    n_correct_labels = 0
    start_time = time.time()

    for batch in batches:
      n_batches += 1
      if n_batches % 100 == 0:
        print("%d" % n_batches, flush=True, end=" ")

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      outputs = self.sess.run(
        [self.num_correct_heads, self.num_correct_labels, self.num_tokens],
        feed_dict)
      n_correct_heads += outputs[0]
      n_correct_labels += outputs[1]
      n_tokens += outputs[2]
      n_sents += batch["batch_size"]

    UAS = n_correct_heads / n_tokens
    LAS = n_correct_labels / n_tokens
    seconds = time.time() - start_time
    self.logger.info("-- {} set".format(data_name))
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, n_sents / seconds))
    self.logger.info("---- UAS:{:>7.2%} ({:>6}/{:>6})".format(
      UAS, n_correct_heads, n_tokens))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, n_correct_labels, n_tokens))
    return LAS

  def eval(self, preprocessor):
    self.logger.info(str(self.cfg))
    raw_data = load_json(self.cfg["data_path"])[:self.cfg["data_size"]]
    _, indexed_data = self._preprocess_input_data(preprocessor)

    #############
    # Main loop #
    #############
    n_tokens = 0
    n_sents = len(indexed_data)
    n_correct_heads = 0
    n_correct_labels = 0
    start_time = time.time()

    print("PREDICTION START")
    for sent_id, (record, indexed_sent) in enumerate(zip(raw_data,
                                                         indexed_data)):
      assert sent_id == record["sent_id"] == indexed_sent["sent_id"]
      if (sent_id + 1) % 100 == 0:
        print("%d" % (sent_id + 1), flush=True, end=" ")

      batch = self.make_one_batch(indexed_sent)
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)

      ##############
      # Prediction #
      ##############
      feed_dict = self._get_feed_dict(batch)
      predicted_heads, predicted_labels = self.sess.run(
        [self.head_predicts, self.label_predicts], feed_dict)

      ###############
      # Add results #
      ###############
      record["predicted_heads"] = [int(head) for head in predicted_heads[0]]
      record["predicted_labels"] = [self.rev_label_dict[label]
                                    for label in predicted_labels[0]]

      #########
      # Count #
      #########
      puncts = None if self.cfg["include_puncts"] else indexed_sent["puncts"]
      n_cur_correct_heads = self._count_correct_heads(
        gold_heads=record["heads"],
        predicted_heads=record["predicted_heads"],
        puncts=puncts
      )
      n_cur_correct_labels = self._count_correct_labels(
        gold_heads=record["heads"],
        predicted_heads=record["predicted_heads"],
        gold_labels=record["labels"],
        predicted_labels=record["predicted_labels"],
        puncts=puncts
      )
      n_correct_heads += n_cur_correct_heads
      n_correct_labels += n_cur_correct_labels
      # -1 means excluding the ROOT node
      n_tokens += batch["seq_len"][0] - 1
      # - np.sum(batch["puncts"]) means excluding punctuations
      if self.cfg["include_puncts"] is False:
        n_tokens -= np.sum(batch["puncts"])

    #################
    # Print results #
    #################
    UAS = n_correct_heads / n_tokens
    LAS = n_correct_labels / n_tokens
    seconds = time.time() - start_time
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, n_sents / seconds))
    self.logger.info("---- UAS:{:>7.2%} ({:>6}/{:>6})".format(
      UAS, n_correct_heads, n_tokens))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, n_correct_labels, n_tokens))

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
                               data_name + ".predicted_heads_and_labels.json")
    write_json(data_path, raw_data)

  def save_class_rep(self):
    path = os.path.join(self.cfg["checkpoint_path"], "class_reps.npz")
    class_reps = [self.sess.run([v])[0]
                  for v in tf.trainable_variables()
                  if "weights" in v.name]
    np.savez_compressed(path, class_reps)


class MultiTaskInstanceBasedModel(LabeledInstanceBasedModel):

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
      tf.float32, shape=[self.label_vocab_size, 2 * self.cfg["num_units"]],
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

  def _build_encoder_op(self):
    self._build_rnn_op()
    self._build_head_and_dep_projection_op()
    self._build_edge_dense_layer_op()
    self._build_anchor_dep_and_head_rep_op()
    self._build_anchor_u_and_l_edge_rep_op()
    self._build_anchor_edge_rep_op()
    self._build_class_dep_and_head_rep_op()
    self._build_class_u_and_l_edge_rep_op()

  def _build_edge_dense_layer_op(self):
    with tf.variable_scope("edge_rep_dense"):
      self.u_edge_rep_dense = tf.get_variable(
        name="u_edge_rep_dense", shape=[self.cfg["num_units"],
                                        self.cfg["num_units"]])
      self.l_edge_rep_dense = tf.get_variable(
        name="l_edge_rep_dense", shape=[self.cfg["num_units"],
                                        self.cfg["num_units"]])

  def _build_anchor_dep_and_head_rep_op(self):
    with tf.variable_scope("anchor_dep_and_head_rep"):
      n_tokens = self.seq_len[0]
      self.anchor_dep_rep = tf.expand_dims(
        self.dep_rep[:self.n_sents, 1:n_tokens], axis=2)
      self.anchor_head_rep = tf.expand_dims(
        self.head_rep[:self.n_sents, :n_tokens], axis=1)

  def _build_anchor_u_and_l_edge_rep_op(self):
    with tf.variable_scope("anchor_u_and_l_edge_rep"):
      edge_reps = self._create_edge_rep(self.anchor_dep_rep,
                                        self.anchor_head_rep)
      # 1D: batch_size, 2D: n_tokens-1, 3D: n_tokens, 4D: dim
      self.u_anchor_edge_reps = tf.tensordot(
        edge_reps, self.u_edge_rep_dense, axes=[-1, -1])
      self.l_anchor_edge_reps = tf.tensordot(
        edge_reps, self.l_edge_rep_dense, axes=[-1, -1])
      print("U & L anchor edge rep shape: {}".format(
        self.u_anchor_edge_reps.get_shape().as_list()))

  def _build_anchor_edge_rep_op(self):
    with tf.variable_scope("anchor_edge_rep"):
      self.anchor_edge_reps = tf.concat([self.u_anchor_edge_reps,
                                         self.l_anchor_edge_reps],
                                        axis=-1)
      print("anchor edge rep shape: {}".format(
        self.anchor_edge_reps.get_shape().as_list()))

  def _build_class_u_and_l_edge_rep_op(self):
    with tf.variable_scope("class_u_and_l_edge_rep"):
      # 1D: n_classes, 2D: dim
      edge_reps = self._create_edge_rep(self.class_dep_rep,
                                        self.class_head_rep)
      self.u_class_edge_reps = tf.tensordot(
        edge_reps, self.u_edge_rep_dense, axes=[-1, -1])
      self.l_class_edge_reps = tf.tensordot(
        edge_reps, self.l_edge_rep_dense, axes=[-1, -1])
      print("U & L class edge rep shape: {}".format(
        self.u_class_edge_reps.get_shape().as_list()))

  def _build_decoder_op(self):
    self._build_train_head_logits_op()
    self._build_train_label_logits_op()
    self._build_head_logits_op()

  def _build_train_head_logits_op(self):
    with tf.name_scope("train_head_logits"):
      logits = self._compute_logits(self.u_anchor_edge_reps,
                                    self.u_class_edge_reps)
      # 1D: batch_size, 2D: n_tokens-1, 3D: n_tokens
      self.train_head_logits = tf.reduce_sum(logits, axis=-1)

  def _build_head_logits_op(self):
    with tf.name_scope("head_logits"):
      neighbor_reps = self.neighbor_reps[:, :self.cfg["num_units"]]
      # 1D: batch_size, 2D: n_tokens-1, 3D: n_tokens, 4D: n_classes
      logits = self._compute_logits(self.u_anchor_edge_reps,
                                    neighbor_reps)
      # 1D: batch_size, 2D: n_tokens-1, 3D: n_tokens
      self.head_logits = tf.reduce_sum(logits, axis=-1)

  def _build_train_label_logits_op(self):
    with tf.name_scope("train_label_logits"):
      n_tokens = self.seq_len[0]
      logits = self._compute_logits(self.l_anchor_edge_reps,
                                    self.l_class_edge_reps)
      # 1D: batch_size * (n_tokens - 1), 2D: n_tokens, 3D: dim
      train_logits = tf.reshape(logits,
                                shape=[self.n_sents * (n_tokens - 1),
                                       n_tokens,
                                       self.label_vocab_size])
      # 1D: batch_size * (n_tokens-1), 2D: 1
      batch_index = tf.reshape(tf.range(self.n_sents * (n_tokens - 1)),
                               shape=[-1, 1])
      # 1D: batch_size * (n_tokens-1), 2D: 1
      head_index = tf.reshape(self.heads, shape=[-1, 1])
      # 1D: batch_size * (n_tokens-1), 2D: 2
      index = tf.concat([batch_index, head_index], axis=-1)
      # 1D: batch_size, 2D: n_tokens-1, 3D: n_classes
      self.train_label_logits = tf.reshape(tf.gather_nd(train_logits, index),
                                           shape=[self.n_sents,
                                                  n_tokens - 1,
                                                  self.label_vocab_size])

  def _build_label_logits_op(self):
    with tf.name_scope("label_logits"):
      n_tokens = self.seq_len[0]
      if self.cfg["sim"] == "dot":
        anchor_edge_reps = self.l_anchor_edge_reps
      else:
        anchor_edge_reps = tf.nn.l2_normalize(self.l_anchor_edge_reps, axis=-1)
      neighbor_reps = self.neighbor_reps[:, self.cfg["num_units"]:]

      # 1D: batch_size, 2D: n_tokens-1, 3D: n_tokens, 4D: n_classes
      # neighbor_reps have already normalized when self.cfg["sim"] == "cos"
      logits = tf.tensordot(anchor_edge_reps, neighbor_reps,
                            axes=[-1, -1])

      # 1D: batch_size * (n_tokens - 1), 2D: n_tokens, 3D: dim
      logits = tf.reshape(logits,
                          shape=[self.n_sents * (n_tokens - 1),
                                 n_tokens, self.label_vocab_size])
      # 1D: batch_size * (n_tokens-1), 2D: 1
      batch_index = tf.reshape(tf.range(self.n_sents * (n_tokens - 1)),
                               shape=[-1, 1])
      # 1D: batch_size * (n_tokens-1), 2D: 1
      head_index = tf.reshape(self.head_predicts, shape=[-1, 1])
      # 1D: batch_size * (n_tokens-1), 2D: 2
      index = tf.concat([batch_index, head_index], axis=-1)
      # 1D: batch_size, 2D: n_tokens-1, 3D: n_classes
      label_logits = tf.reshape(tf.gather_nd(logits, index),
                                shape=[self.n_sents,
                                       n_tokens - 1,
                                       self.label_vocab_size])
      self.label_logits = label_logits / self.n_each_label

  def _build_loss_op(self):
    self._build_head_loss_op()
    self._build_label_loss_op()
    self.loss = self.head_loss + self.label_loss
    tf.summary.scalar("loss", self.loss)

  def _build_head_loss_op(self):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.train_head_logits, labels=self.heads)
    self.head_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=-1))
    tf.summary.scalar("head_loss", self.head_loss)

  def _build_label_loss_op(self):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.train_label_logits, labels=self.labels)
    self.label_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=-1))
    tf.summary.scalar("label_loss", self.label_loss)

  def _build_predict_op(self):
    self._build_train_head_prediction_op()
    self._build_train_label_prediction_op()
    self._build_head_prediction_op()
    self._build_label_logits_op()
    self._build_label_prediction_op()

  def _build_train_head_prediction_op(self):
    self.train_head_predicts = tf.cast(
      tf.argmax(self.train_head_logits, axis=-1), tf.int32)

  def _build_train_label_prediction_op(self):
    self.train_label_predicts = tf.cast(
      tf.argmax(self.train_label_logits, axis=-1), tf.int32)

  def _build_head_prediction_op(self):
    self.head_predicts = tf.cast(
      tf.argmax(self.head_logits, axis=-1), tf.int32)

  def _build_label_prediction_op(self):
    self.label_predicts = tf.cast(
      tf.argmax(self.label_logits, axis=-1), tf.int32)

  def _build_answer_check_op(self):
    self._build_count_correct_train_heads_op()
    self._build_count_correct_train_labels_op()
    self._build_count_correct_heads_op()
    self._build_count_correct_labels_op()

  def _build_count_correct_train_heads_op(self):
    correct_heads = tf.cast(
      tf.equal(self.train_head_predicts, self.heads), tf.int32)
    n_tokens = (self.seq_len[0] - 1) * self.n_sents
    if self.cfg["include_puncts"]:
      self.correct_train_heads = correct_heads
      self.num_train_tokens = n_tokens
    else:
      not_puncts = 1 - self.puncts
      self.correct_train_heads = correct_heads * not_puncts
      self.num_train_tokens = n_tokens - tf.reduce_sum(self.puncts)
    self.num_correct_train_heads = tf.reduce_sum(self.correct_train_heads)

  def _build_count_correct_train_labels_op(self):
    correct_labels = tf.cast(
      tf.equal(self.train_label_predicts, self.labels), tf.int32)
    self.correct_train_labels = correct_labels * self.correct_train_heads
    self.num_correct_train_labels = tf.reduce_sum(self.correct_train_labels)

  def _build_count_correct_labels_op(self):
    correct_labels = tf.cast(
      tf.equal(self.label_predicts, self.labels), tf.int32)
    self.correct_labels = correct_labels * self.correct_heads
    self.num_correct_labels = tf.reduce_sum(self.correct_labels)

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

      for sent_heads, sent_labels, sent_edge_reps in zip(
          batch["heads"], batch["labels"], edge_reps):
        for h, r, reps in zip(sent_heads, sent_labels, sent_edge_reps):
          rep = reps[h]
          if self.sim == "cos":
            rep = l2_normalize(rep)
          precomp_edge_reps[r] += rep
    self.precomputed_edge_rep = precomp_edge_reps
    seconds = time.time() - start_time
    self.logger.info("------ Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds))

  def train_epoch(self, batches):
    loss_total = 0.
    n_tokens = 0
    n_sents = 0
    n_batches = 0
    n_correct_heads = 0
    n_correct_labels = 0
    start_time = time.time()

    for batch in batches:
      n_batches += 1
      if n_batches % 100 == 0:
        print("%d" % n_batches, flush=True, end=" ")

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=True)
      feed_dict = self._get_feed_dict(
        batch, is_train=True, keep_prob=self.cfg["keep_prob"],
        lr=self.cfg["lr"])
      outputs = self.sess.run(
        [self.train_op, self.loss, self.num_correct_train_heads,
         self.num_correct_train_labels, self.num_train_tokens], feed_dict)

      loss_total += outputs[1]
      n_correct_heads += outputs[2]
      n_correct_labels += outputs[3]
      n_tokens += outputs[4]
      n_sents += batch["n_sents"]

    avg_loss = loss_total / n_batches
    UAS = n_correct_heads / n_tokens
    LAS = n_correct_labels / n_tokens
    seconds = time.time() - start_time
    self.logger.info("-- Train set")
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, n_sents / seconds))
    self.logger.info("---- Loss: {:.2f} ({:.2f}/{:d})".format(
      avg_loss, loss_total, n_batches))
    self.logger.info("---- UAS:{:>7.2%} ({:>6}/{:>6})".format(
      UAS, n_correct_heads, n_tokens))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, n_correct_labels, n_tokens))
    return avg_loss, loss_total

  def evaluate_epoch(self, batches, data_name):
    if self.use_all_instances:
      self.precompute_edge_rep(dim_rep=self.cfg["num_units"] * 2)
    else:
      self.precompute_edge_rep_for_random_sampling()

    n_tokens = 0
    n_sents = 0
    n_batches = 0
    n_correct_heads = 0
    n_correct_labels = 0
    start_time = time.time()

    for batch in batches:
      n_batches += 1
      if n_batches % 100 == 0:
        print("%d" % n_batches, flush=True, end=" ")

      batch = self.add_precomputed_rep_to_batch(batch)
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      outputs = self.sess.run(
        [self.num_correct_heads, self.num_correct_labels, self.num_tokens],
        feed_dict)
      n_correct_heads += outputs[0]
      n_correct_labels += outputs[1]
      n_tokens += outputs[2]
      n_sents += batch["n_sents"]

    UAS = n_correct_heads / n_tokens
    LAS = n_correct_labels / n_tokens
    seconds = time.time() - start_time
    self.logger.info("-- {} set".format(data_name))
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, n_sents / seconds))
    self.logger.info("---- UAS:{:>7.2%} ({:>6}/{:>6})".format(
      UAS, n_correct_heads, n_tokens))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, n_correct_labels, n_tokens))
    return LAS

  def eval(self, preprocessor):
    self.logger.info(str(self.cfg))
    raw_data = load_json(self.cfg["data_path"])[:self.cfg["data_size"]]
    _, indexed_data = self._preprocess_input_data(preprocessor)

    ################################################
    # Precomputing edge reps of training instances #
    ################################################
    if self.use_all_instances:
      self.precompute_edge_rep(dim_rep=self.cfg["num_units"] * 2)
    else:
      self.precompute_edge_rep_for_random_sampling()

    #############
    # Main loop #
    #############
    n_tokens = 0
    n_sents = len(indexed_data)
    n_correct_heads = 0
    n_correct_labels = 0
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

      ##############
      # Prediction #
      ##############
      feed_dict = self._get_feed_dict(batch)
      predicted_heads, predicted_labels = self.sess.run(
        [self.head_predicts, self.label_predicts], feed_dict)

      ###############
      # Add results #
      ###############
      record["predicted_heads"] = [int(head) for head in predicted_heads[0]]
      record["predicted_labels"] = [self.rev_label_dict[label]
                                    for label in predicted_labels[0]]

      #########
      # Count #
      #########
      puncts = None if self.cfg["include_puncts"] else indexed_sent["puncts"]
      n_cur_correct_heads = self._count_correct_heads(
        gold_heads=record["heads"],
        predicted_heads=record["predicted_heads"],
        puncts=puncts
      )
      n_cur_correct_labels = self._count_correct_labels(
        gold_heads=record["heads"],
        predicted_heads=record["predicted_heads"],
        gold_labels=record["labels"],
        predicted_labels=record["predicted_labels"],
        puncts=puncts
      )
      n_correct_heads += n_cur_correct_heads
      n_correct_labels += n_cur_correct_labels
      # -1 means excluding the ROOT node
      n_tokens += batch["seq_len"][0] - 1
      # - np.sum(batch["puncts"]) means excluding punctuations
      if self.cfg["include_puncts"] is False:
        n_tokens -= np.sum(batch["puncts"])

    #################
    # Print results #
    #################
    UAS = n_correct_heads / n_tokens
    LAS = n_correct_labels / n_tokens
    seconds = time.time() - start_time
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, n_sents / seconds))
    self.logger.info("---- UAS:{:>7.2%} ({:>6}/{:>6})".format(
      UAS, n_correct_heads, n_tokens))
    self.logger.info("---- LAS:{:>7.2%} ({:>6}/{:>6})".format(
      LAS, n_correct_labels, n_tokens))

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
                               data_name + ".predicted_heads_and_labels.json")
    write_json(data_path, raw_data)


class HingeLossLabeler(MultiTaskInstanceBasedModel):

  def _build_train_logits_op(self):
    with tf.name_scope("train_logits"):
      if self.cfg["sim"] == "dot":
        head_logits = tf.tensordot(
          self.u_anchor_edge_reps, self.u_class_edge_reps, axes=[-1, -1])
        label_logits = tf.tensordot(
          self.l_anchor_edge_reps, self.l_class_edge_reps, axes=[-1, -1])
      else:
        head_logits = tf.tensordot(
          tf.nn.l2_normalize(self.u_anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.u_class_edge_reps, axis=-1),
          axes=[-1, -1]) * self.cfg["scaling_factor"]
        label_logits = tf.tensordot(
          tf.nn.l2_normalize(self.l_anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.l_class_edge_reps, axis=-1),
          axes=[-1, -1]) * self.cfg["scaling_factor"]

      # 1D: batch_size, 2D: num_tokens-1, 3D: num_tokens, 4D: num_classes
      train_u_logits = tf.reshape(
        head_logits, shape=[self.n_sents, self.seq_len[0] - 1,
                            self.seq_len[0], self.label_vocab_size])
      train_l_logits = tf.reshape(
        label_logits, shape=[self.n_sents, self.seq_len[0] - 1,
                             self.seq_len[0], self.label_vocab_size])
      self.train_logits = train_u_logits + train_l_logits

  def _build_train_gold_logits_op(self):
    with tf.name_scope("train_gold_logits"):
      num_tokens = self.seq_len[0]
      # 1D: batch_size * (num_tokens - 1), 2D: num_tokens, 3D: num_classes
      train_logits = tf.reshape(self.train_logits,
                                shape=[self.n_sents * (num_tokens - 1),
                                       num_tokens,
                                       self.label_vocab_size])
      # 1D: batch_size * (num_tokens-1), 2D: 1
      batch_index = tf.reshape(tf.range(self.n_sents * (num_tokens - 1)),
                               shape=[-1, 1])
      # 1D: batch_size * (num_tokens-1), 2D: 1
      head_index = tf.reshape(self.heads, shape=[-1, 1])
      # 1D: batch_size * (num_tokens-1), 2D: 2
      index = tf.concat([batch_index, head_index], axis=-1)
      # 1D: batch_size * (num_tokens-1), 2D: num_classes
      gold_head_logits = tf.gather_nd(train_logits, index)

      # 1D: batch_size * (num_tokens-1), 2D: 1
      label_index = tf.reshape(self.labels, shape=[-1, 1])
      # 1D: batch_size * (num_tokens-1), 2D: 2
      index = tf.concat([batch_index, label_index], axis=-1)
      # 1D: batch_size * (num_tokens-1)
      gold_logits = tf.gather_nd(gold_head_logits, index)
      # 1D: batch_size, 2D: num_tokens - 1
      self.train_gold_logits = tf.reshape(gold_logits,
                                          shape=[self.n_sents, num_tokens - 1])

  def _build_train_pred_logits_op(self):
    with tf.name_scope("train_pred_logits"):
      # 1D: batch_size, 2D: num_tokens - 1, 3D: num_tokens
      pred_label_logits = tf.reduce_max(self.train_logits, axis=-1)
      # 1D: batch_size, 2D: num_tokens - 1
      self.train_pred_logits = tf.reduce_max(pred_label_logits, axis=-1)

  def _build_loss_op(self):
    # 1D: batch_size
    gold_logits = tf.reduce_sum(self.train_gold_logits, axis=-1)
    pred_logits = tf.reduce_sum(self.train_pred_logits, axis=-1)
    losses = tf.maximum(0, 1 + pred_logits - gold_logits)
    self.loss = tf.reduce_mean(losses)
    tf.summary.scalar("loss", self.loss)

  def _build_logits_op(self):
    with tf.name_scope("logits"):
      u_edge_reps = self.edge_reps[:, :, :, :self.cfg["num_units"]]
      l_edge_reps = self.edge_reps[:, :, :, self.cfg["num_units"]:]
      u_neighbor_reps = self.neighbor_reps[:, :self.cfg["num_units"]]
      l_neighbor_reps = self.neighbor_reps[:, self.cfg["num_units"]:]

      # 1D: batch_size, 2D: num_tokens-1, 3D: num_tokens, 4D: num_classes
      u_logits = tf.tensordot(u_edge_reps, u_neighbor_reps, axes=[-1, -1])
      l_logits = tf.tensordot(l_edge_reps, l_neighbor_reps, axes=[-1, -1])

      # 1D: batch_size, 2D: num_tokens-1, 3D: num_tokens, 4D: num_classes
      u_logits = tf.reshape(
        u_logits, shape=[self.n_sents, self.seq_len[0] - 1,
                         self.seq_len[0], self.label_vocab_size])
      # 1D: batch_size, 2D: num_tokens-1, 3D: num_tokens, 4D: num_classes
      l_logits = tf.reshape(
        l_logits, shape=[self.n_sents, self.seq_len[0] - 1,
                         self.seq_len[0], self.label_vocab_size])
      self.logits = u_logits + l_logits

  def _build_head_prediction_op(self):
    with tf.name_scope("head_predicts"):
      # 1D: batch_size, 2D: num_tokens - 1
      self.head_predicts = tf.cast(tf.argmax(
        tf.reduce_max(self.logits, axis=-1), axis=-1), dtype=tf.int32)

  def _build_label_prediction_op(self):
    with tf.name_scope("label_predicts"):
      num_tokens = self.seq_len[0]
      # 1D: batch_size * (num_tokens - 1), 2D: num_tokens, 3D: num_classes
      logits = tf.reshape(self.logits,
                          shape=[self.n_sents * (num_tokens - 1),
                                 num_tokens,
                                 self.label_vocab_size])
      # 1D: batch_size * (num_tokens-1), 2D: 1
      batch_index = tf.reshape(tf.range(self.n_sents * (num_tokens - 1)),
                               shape=[-1, 1])
      # 1D: batch_size * (num_tokens-1), 2D: 1
      head_index = tf.reshape(self.head_predicts, shape=[-1, 1])
      # 1D: batch_size * (num_tokens-1), 2D: 2
      index = tf.concat([batch_index, head_index], axis=-1)
      # 1D: batch_size * (num_tokens-1), 2D: num_classes
      label_logits = tf.gather_nd(logits, index)
      # 1D: batch_size, 2D: (num_tokens-1), 3D: num_classes
      label_logits = tf.reshape(label_logits, shape=[self.n_sents,
                                                     num_tokens - 1,
                                                     self.label_vocab_size])
      # 1D: batch_size, 2D: (num_tokens-1)
      self.label_predicts = tf.cast(tf.argmax(label_logits, axis=-1),
                                    dtype=tf.int32)

  def _build_encoder_op(self):
    self._build_rnn_op()
    self._build_head_and_dep_projection_op()
    self._build_edge_rep_op()
    self._build_train_edge_rep_op()
    self._build_class_dep_and_head_rep_op()
    self._build_class_edge_rep_op()

  def _build_decoder_op(self):
    self._build_train_logits_op()
    self._build_train_gold_logits_op()
    self._build_train_pred_logits_op()
    self._build_logits_op()

  def _build_predict_op(self):
    self._build_head_prediction_op()
    self._build_label_prediction_op()

  def _build_answer_check_op(self):
    self._build_count_correct_heads_op()
    self._build_count_correct_labels_op()

  def train_epoch(self, batches):
    loss_total = 0.
    num_batches = 0
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
      _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict)
      loss_total += train_loss

    avg_loss = loss_total / num_batches
    seconds = time.time() - start_time
    self.logger.info("-- Train set")
    self.logger.info("---- Time: {:.2f} sec".format(seconds))
    self.logger.info("---- Loss: {:.2f} ({:.2f}/{:d})".format(
      avg_loss, loss_total, num_batches))
    return avg_loss, loss_total
