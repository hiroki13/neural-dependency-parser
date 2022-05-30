# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time

import numpy as np
import tensorflow as tf

from models.labeled_models import LabeledWeightBasedModel
from utils.common import load_json, write_json


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
