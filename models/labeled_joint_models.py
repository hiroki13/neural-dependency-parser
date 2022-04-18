# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from models.labeled_models import OnlineSoftmaxInstanceLossLabeler


class JointLossJointLogitLabeler(OnlineSoftmaxInstanceLossLabeler):

  def _build_train_logits_op(self):
    with tf.name_scope("train_logits"):
      self.label_weights = tf.get_variable(
        name="label_weights", trainable=True,
        shape=[self.label_vocab_size, self.cfg["num_units"]])

      if self.cfg["sim"] == "dot":
        c_logits = tf.matmul(
          self.anchor_edge_reps, self.label_weights, transpose_b=True)
        i_logits = tf.matmul(
          self.anchor_edge_reps, self.class_edge_reps, transpose_b=True)
      else:
        c_logits = tf.matmul(
          tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.label_weights, axis=-1),
          transpose_b=True) * self.cfg["scaling_factor"]
        i_logits = tf.matmul(
          tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.class_edge_reps, axis=-1),
          transpose_b=True) * self.cfg["scaling_factor"]

      # 1D: batch_size, 2D: num_tokens-1, 3D: num_classes
      self.train_c_logits = tf.reshape(c_logits, shape=[self.batch_size,
                                                        self.seq_len[0] - 1,
                                                        self.label_vocab_size])
      self.train_i_logits = tf.reshape(i_logits, shape=[self.batch_size,
                                                        self.seq_len[0] - 1,
                                                        self.label_vocab_size])
      self.train_logits = self.train_c_logits + self.train_i_logits

  def _build_logits_op(self):
    with tf.name_scope("logits"):
      if self.cfg["sim"] == "dot":
        anchor_edge_reps = self.anchor_edge_reps
      else:
        anchor_edge_reps = tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1)
      # 1D: batch_size * (num_tokens-1), 2D: num_classes
      c_logits = tf.matmul(
        anchor_edge_reps, self.label_weights, transpose_b=True)
      i_logits = tf.matmul(
        anchor_edge_reps, self.neighbor_reps, transpose_b=True)
      # 1D: batch_size, 2D: num_tokens-1, 3D: num_classes
      c_logits = tf.reshape(c_logits, shape=[self.batch_size,
                                             self.seq_len[0] - 1,
                                             self.label_vocab_size])
      i_logits = tf.reshape(i_logits, shape=[self.batch_size,
                                             self.seq_len[0] - 1,
                                             self.label_vocab_size])
      self.logits = c_logits + i_logits

  def _build_label_loss_op(self):
    c_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.train_c_logits, labels=self.labels)
    i_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.train_i_logits, labels=self.labels)
    c_loss = tf.reduce_mean(tf.reduce_sum(c_losses, axis=-1))
    i_loss = tf.reduce_mean(tf.reduce_sum(i_losses, axis=-1))
    alpha = self.cfg["alpha"]
    self.label_loss = alpha * c_loss + (1 - alpha) * i_loss


class JointLossJointSoftmaxLabeler(JointLossJointLogitLabeler):

  def _build_train_logits_op(self):
    with tf.name_scope("train_logits"):
      self.label_weights = tf.get_variable(
        name="label_weights", trainable=True,
        shape=[self.label_vocab_size, self.cfg["num_units"]])

      if self.cfg["sim"] == "dot":
        c_logits = tf.matmul(
          self.anchor_edge_reps, self.label_weights, transpose_b=True)
        i_logits = tf.matmul(
          self.anchor_edge_reps, self.class_edge_reps, transpose_b=True)
      else:
        c_logits = tf.matmul(
          tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.label_weights, axis=-1),
          transpose_b=True) * self.cfg["scaling_factor"]
        i_logits = tf.matmul(
          tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.class_edge_reps, axis=-1),
          transpose_b=True) * self.cfg["scaling_factor"]

      # 1D: batch_size, 2D: num_tokens-1, 3D: num_classes
      self.train_c_logits = tf.reshape(c_logits, shape=[self.batch_size,
                                                        self.seq_len[0] - 1,
                                                        self.label_vocab_size])
      self.train_i_logits = tf.reshape(i_logits, shape=[self.batch_size,
                                                        self.seq_len[0] - 1,
                                                        self.label_vocab_size])
      self.train_logits = tf.nn.softmax(self.train_c_logits, axis=-1) + \
                          tf.nn.softmax(self.train_i_logits, axis=-1)

  def _build_logits_op(self):
    with tf.name_scope("logits"):
      if self.cfg["sim"] == "dot":
        anchor_edge_reps = self.anchor_edge_reps
      else:
        anchor_edge_reps = tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1)
      # 1D: batch_size * (num_tokens-1), 2D: num_classes
      c_logits = tf.matmul(
        anchor_edge_reps, self.label_weights, transpose_b=True)
      i_logits = tf.matmul(
        anchor_edge_reps, self.neighbor_reps, transpose_b=True)
      # 1D: batch_size, 2D: num_tokens-1, 3D: num_classes
      c_logits = tf.reshape(c_logits, shape=[self.batch_size,
                                             self.seq_len[0] - 1,
                                             self.label_vocab_size])
      i_logits = tf.reshape(i_logits, shape=[self.batch_size,
                                             self.seq_len[0] - 1,
                                             self.label_vocab_size])
      self.logits = tf.nn.softmax(c_logits, axis=-1) + \
                    tf.nn.softmax(i_logits, axis=-1)


class JointLossOnlyClassifierLabeler(JointLossJointLogitLabeler):

  def _build_train_logits_op(self):
    with tf.name_scope("train_logits"):
      self.label_weights = tf.get_variable(
        name="label_weights", trainable=True,
        shape=[self.label_vocab_size, self.cfg["num_units"]])

      if self.cfg["sim"] == "dot":
        c_logits = tf.matmul(
          self.anchor_edge_reps,
          self.label_weights,
          transpose_b=True)
        i_logits = tf.matmul(
          self.anchor_edge_reps,
          self.class_edge_reps,
          transpose_b=True)
      else:
        c_logits = tf.matmul(
          tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.label_weights, axis=-1),
          transpose_b=True) * self.cfg["scaling_factor"]
        i_logits = tf.matmul(
          tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.class_edge_reps, axis=-1),
          transpose_b=True) * self.cfg["scaling_factor"]

      # 1D: batch_size, 2D: num_tokens-1, 3D: num_classes
      self.train_c_logits = tf.reshape(c_logits, shape=[self.batch_size,
                                                        self.seq_len[0] - 1,
                                                        self.label_vocab_size])
      self.train_i_logits = tf.reshape(i_logits, shape=[self.batch_size,
                                                        self.seq_len[0] - 1,
                                                        self.label_vocab_size])
      self.train_logits = self.train_c_logits

  def _build_logits_op(self):
    with tf.name_scope("logits"):
      if self.cfg["sim"] == "dot":
        anchor_edge_reps = self.anchor_edge_reps
      else:
        anchor_edge_reps = tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1)
      # 1D: batch_size * (num_tokens-1), 2D: num_classes
      logits = tf.matmul(
        anchor_edge_reps, self.label_weights, transpose_b=True)
      # 1D: batch_size, 2D: num_tokens-1, 3D: num_classes
      self.logits = tf.reshape(logits, shape=[self.batch_size,
                                              self.seq_len[0] - 1,
                                              self.label_vocab_size])


class JointLossOnlyInstanceLabeler(JointLossJointLogitLabeler):

  def _build_train_logits_op(self):
    with tf.name_scope("train_logits"):
      self.label_weights = tf.get_variable(
        name="label_weights", trainable=True,
        shape=[self.label_vocab_size, self.cfg["num_units"]])

      if self.cfg["sim"] == "dot":
        c_logits = tf.matmul(
          self.anchor_edge_reps,
          self.label_weights,
          transpose_b=True)
        i_logits = tf.matmul(
          self.anchor_edge_reps,
          self.class_edge_reps,
          transpose_b=True)
      else:
        c_logits = tf.matmul(
          tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.label_weights, axis=-1),
          transpose_b=True) * self.cfg["scaling_factor"]
        i_logits = tf.matmul(
          tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1),
          tf.nn.l2_normalize(self.class_edge_reps, axis=-1),
          transpose_b=True) * self.cfg["scaling_factor"]

      # 1D: batch_size, 2D: num_tokens-1, 3D: num_classes
      self.train_c_logits = tf.reshape(c_logits, shape=[self.batch_size,
                                                        self.seq_len[0] - 1,
                                                        self.label_vocab_size])
      self.train_i_logits = tf.reshape(i_logits, shape=[self.batch_size,
                                                        self.seq_len[0] - 1,
                                                        self.label_vocab_size])
      self.train_logits = self.train_i_logits

  def _build_logits_op(self):
    with tf.name_scope("logits"):
      if self.cfg["sim"] == "dot":
        anchor_edge_reps = self.anchor_edge_reps
      else:
        anchor_edge_reps = tf.nn.l2_normalize(self.anchor_edge_reps, axis=-1)
      # 1D: batch_size * (num_tokens-1), 2D: num_classes
      logits = tf.matmul(
        anchor_edge_reps, self.neighbor_reps, transpose_b=True)
      # 1D: batch_size, 2D: num_tokens-1, 3D: num_classes
      self.logits = tf.reshape(logits, shape=[self.batch_size,
                                              self.seq_len[0] - 1,
                                              self.label_vocab_size])
