# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from models.unlabeled_models import UnlabeledWeightBasedModel, \
  UnlabeledInstanceBasedModel


class SigmoidCrossEntropyModel(UnlabeledWeightBasedModel):

  def _add_placeholders(self):
    self.words = tf.placeholder(tf.int32,
                                shape=[None, None],
                                name="words")
    self.pos_tags = tf.placeholder(tf.int32,
                                   shape=[None, None],
                                   name="pos_tags")
    self.heads = tf.placeholder(tf.int32,
                                shape=[None, None],
                                name="heads")
    self.head_one_hots = tf.placeholder(tf.float32,
                                        shape=[None, None, None],
                                        name="head_one_hots")
    self.puncts = tf.placeholder(tf.int32,
                                 shape=[None, None],
                                 name="puncts")
    self.seq_len = tf.placeholder(tf.int32,
                                  shape=[None],
                                  name="seq_len")
    if self.cfg["use_chars"]:
      self.chars = tf.placeholder(tf.int32,
                                  shape=[None, None, None],
                                  name="chars")
    # hyperparameters
    self.is_train = tf.placeholder(tf.bool, name="is_train")
    self.keep_prob = tf.placeholder(tf.float32, name="rnn_keep_probability")
    self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
    self.lr = tf.placeholder(tf.float32, name="learning_rate")

  def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
    feed_dict = {self.words: batch["words"],
                 self.pos_tags: batch["pos_tags"],
                 self.puncts: batch["puncts"],
                 self.seq_len: batch["seq_len"]}

    if self.cfg["use_chars"]:
      feed_dict[self.chars] = batch["chars"]
    if "heads" in batch:
      feed_dict[self.heads] = batch["heads"]
    if "head_one_hots" in batch:
      feed_dict[self.head_one_hots] = batch["head_one_hots"]

    feed_dict[self.keep_prob] = keep_prob
    feed_dict[self.drop_rate] = 1.0 - keep_prob
    feed_dict[self.is_train] = is_train
    if lr is not None:
      feed_dict[self.lr] = lr

    return feed_dict

  def _build_head_loss_op(self):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
      logits=self.head_logits, labels=self.head_one_hots)
    self.head_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]))
    tf.summary.scalar("head_loss", self.head_loss)


class OnlineSigmoidInstanceLossModel(UnlabeledInstanceBasedModel):

  def _add_placeholders(self):
    self.words = tf.placeholder(tf.int32,
                                shape=[None, None],
                                name="words")
    self.pos_tags = tf.placeholder(tf.int32,
                                   shape=[None, None],
                                   name="pos_tags")
    self.heads = tf.placeholder(tf.int32,
                                shape=[None, None],
                                name="heads")
    self.puncts = tf.placeholder(tf.int32,
                                 shape=[None, None],
                                 name="puncts")
    self.seq_len = tf.placeholder(tf.int32,
                                  shape=[None],
                                  name="seq_len")
    self.batch_size = tf.placeholder(tf.int32,
                                     shape=None,
                                     name="batch_size")
    self.neighbor_deps = tf.placeholder(tf.int32,
                                     shape=None,
                                     name="neighbor_dep_indices")
    self.neighbor_heads = tf.placeholder(tf.int32,
                                      shape=None,
                                      name="neighbor_head_indices")
    self.head_one_hots = tf.placeholder(tf.float32,
                                        shape=[None, None, None],
                                        name="head_one_hots")
    self.head_loss_masks = tf.placeholder(tf.float32,
                                          shape=[None, None, None],
                                          name="head_loss_masks")
    if self.cfg["use_chars"]:
      self.chars = tf.placeholder(tf.int32,
                                  shape=[None, None, None],
                                  name="chars")
    # hyperparameters
    self.is_train = tf.placeholder(tf.bool, name="is_train")
    self.keep_prob = tf.placeholder(tf.float32, name="rnn_keep_probability")
    self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
    self.lr = tf.placeholder(tf.float32, name="learning_rate")

  def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
    feed_dict = {self.words: batch["words"],
                 self.pos_tags: batch["pos_tags"],
                 self.puncts: batch["puncts"],
                 self.seq_len: batch["seq_len"],
                 self.batch_size: batch["batch_size"],
                 self.neighbor_deps: batch["neighbor_deps"],
                 self.neighbor_heads:  batch["neighbor_heads"]}
    if self.cfg["use_chars"]:
      feed_dict[self.chars] = batch["chars"]
    if "heads" in batch:
      feed_dict[self.heads] = batch["heads"]
    if "head_one_hots" in batch:
      feed_dict[self.head_one_hots] = batch["head_one_hots"]
    if "head_loss_masks" in batch:
      feed_dict[self.head_loss_masks] = batch["head_loss_masks"]
    feed_dict[self.keep_prob] = keep_prob
    feed_dict[self.drop_rate] = 1.0 - keep_prob
    feed_dict[self.is_train] = is_train
    if lr is not None:
      feed_dict[self.lr] = lr
    return feed_dict

  def _build_head_loss_op(self):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
      logits=self.head_logits, labels=self.head_one_hots)
    if self.cfg["num_negatives"] > 0:
      losses = losses * self.head_loss_masks
    self.head_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]))
    tf.summary.scalar("head_loss", self.head_loss)
