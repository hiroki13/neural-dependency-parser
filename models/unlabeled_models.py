# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import h5py
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn.python.ops.rnn import \
  stack_bidirectional_dynamic_rnn

from models.base_models import BaseModel
from models.network_components import multi_conv1d, highway_network
from utils.common import load_json, write_json

VALID_BATCH_SIZE = 128


class UnlabeledWeightBasedModel(BaseModel):

  def __init__(self, config, batcher, is_train=True):
    super(UnlabeledWeightBasedModel, self).__init__(config, batcher, is_train)
    if config["use_words"] or config["use_bert"] is False:
      w_init = np.load(self.cfg["pretrained_emb"])["embeddings"]
      w_place_holder = tf.placeholder(
        dtype=tf.float32, shape=[len(w_init), len(w_init[0])])
      self.sess.run(
        self.token_emb.assign(w_place_holder), {w_place_holder: w_init})
    if is_train:
      self._build_answer_check_op()

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

    feed_dict[self.keep_prob] = keep_prob
    feed_dict[self.drop_rate] = 1.0 - keep_prob
    if self.cfg["use_bert"]:
      feed_dict[self.bert_drop_rate] = 1.0 - self.cfg["bert_keep_prob"]
    feed_dict[self.is_train] = is_train
    if lr is not None:
      feed_dict[self.lr] = lr

    return feed_dict

  def _build_embedding_op(self):
    with tf.variable_scope("embeddings"):
      #############
      # Token emb #
      #############
      if self.cfg["use_bert"]:
        root_emb = tf.get_variable(
          name="root_emb", dtype=tf.float32, trainable=True,
          shape=[1, self.cfg["bert_dim"]])
        ones = tf.ones(shape=[self.batch_size, 1, 1], dtype=tf.float32)
        root_emb = ones * root_emb
        bert_rep = tf.concat([root_emb, self.bert_rep], axis=1)
        bert_rep = tf.layers.dropout(
          bert_rep, rate=self.bert_drop_rate, training=self.is_train)
      else:
        bert_rep = None

      ############
      # Word emb #
      ############
      if self.cfg["use_words"] or self.cfg["use_bert"] is False:
        padding_token_emb = tf.get_variable(
          name="padding_emb", dtype=tf.float32, trainable=False,
          shape=[1, self.cfg["emb_dim"]])
        special_token_emb = tf.get_variable(
          name="spacial_emb", dtype=tf.float32, trainable=True,
          shape=[3, self.cfg["emb_dim"]])
        self.token_emb = tf.get_variable(
          name="emb", dtype=tf.float32, trainable=self.cfg["tuning_emb"],
          shape=[self.cfg["vocab_size"], self.cfg["emb_dim"]])
        self.word_embeddings = tf.concat(
          [padding_token_emb, special_token_emb, self.token_emb], axis=0)
        word_emb = tf.nn.embedding_lookup(
          self.word_embeddings, self.words, name="words_emb")
      else:
        word_emb = None
      print("word embedding shape: {}".format(word_emb.get_shape().as_list()))

      #################
      # Character emb #
      #################
      if self.cfg["use_chars"]:
        self.char_embeddings = tf.get_variable(
          name="char_emb", dtype=tf.float32, trainable=True,
          shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
        char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars,
                                          name="chars_emb")
        char_rep = multi_conv1d(
          char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
          drop_rate=self.drop_rate, is_train=self.is_train)
        print("chars representation shape: {}".format(
          char_rep.get_shape().as_list()))
      else:
        char_rep = None

      ###############
      # POS tag emb #
      ###############
      if self.cfg["use_pos_tags"]:
        self.pos_embeddings = tf.get_variable(
          name="pos_emb", dtype=tf.float32, trainable=True,
          shape=[self.pos_vocab_size, self.cfg["pos_emb_dim"]])
        pos_emb = tf.nn.embedding_lookup(
          self.pos_embeddings, self.pos_tags, name="pos_emb")
        print("pos embedding shape: {}".format(pos_emb.get_shape().as_list()))
      else:
        pos_emb = None

      #################
      # Concatenation #
      #################
      concat_emb_list = []
      if word_emb is not None:
        concat_emb_list.append(word_emb)
      if char_rep is not None:
        concat_emb_list.append(char_rep)
      if pos_emb is not None:
        concat_emb_list.append(pos_emb)
      concat_emb = tf.concat(concat_emb_list, axis=-1)

      ##############################
      # Transform concatenated emb #
      ##############################
      if self.cfg["use_highway"]:
        concat_rep = highway_network(
          concat_emb, self.cfg["highway_layers"], use_bias=True,
          bias_init=0.0, keep_prob=self.keep_prob, is_train=self.is_train)
      else:
        concat_rep = tf.layers.dropout(
          concat_emb, rate=self.drop_rate, training=self.is_train)
      if bert_rep is not None:
        self.word_emb = tf.concat([bert_rep, concat_rep], axis=-1)
      else:
        self.word_emb = concat_emb

      print("transformed input emb shape: {}".format(
        self.word_emb.get_shape().as_list()))

  def _build_encoder_op(self):
    self._build_rnn_op()
    self._build_head_and_dep_projection_op()
    self._build_edge_dense_layer_op()
    self._build_edge_rep_op()

  def _build_rnn_op(self):
    with tf.variable_scope("bi_directional_rnn"):
      cell_fw = self._create_rnn_cell()
      cell_bw = self._create_rnn_cell()

      if self.cfg["use_stack_rnn"]:
        rnn_outs, *_ = stack_bidirectional_dynamic_rnn(
          cell_fw, cell_bw, self.word_emb,
          dtype=tf.float32, sequence_length=self.seq_len)
      else:
        rnn_outs, *_ = bidirectional_dynamic_rnn(
          cell_fw, cell_bw, self.word_emb,
          dtype=tf.float32, sequence_length=self.seq_len)
      rnn_outs = tf.concat(rnn_outs, axis=-1)
      rnn_outs = tf.layers.dropout(rnn_outs,
                                   rate=self.drop_rate,
                                   training=self.is_train)
      self.rnn_outs = rnn_outs
      print("rnn output shape: {}".format(rnn_outs.get_shape().as_list()))

  def _create_rnn_cell(self):
    if self.cfg["num_layers"] is None or self.cfg["num_layers"] <= 1:
      return self._create_single_rnn_cell(self.cfg["num_units"])
    else:
      if self.cfg["use_stack_rnn"]:
        lstm_cells = []
        for i in range(self.cfg["num_layers"]):
          cell = tf.nn.rnn_cell.LSTMCell(
            self.cfg["num_units"], initializer=tf.initializers.orthogonal)
          cell = tf.contrib.rnn.DropoutWrapper(
            cell, state_keep_prob=self.keep_prob,
            input_keep_prob=self.keep_prob, dtype=tf.float32)
          lstm_cells.append(cell)
        return lstm_cells
      else:
        return MultiRNNCell(
          [self._create_single_rnn_cell(self.cfg["num_units"])
           for _ in range(self.cfg["num_layers"])])

  def _build_head_and_dep_projection_op(self):
    with tf.variable_scope("head_and_dep_projection"):
      head_rep = tf.layers.dense(self.rnn_outs,
                                 activation=tf.nn.relu,
                                 units=self.cfg["num_units"],
                                 use_bias=True)
      dep_rep = tf.layers.dense(self.rnn_outs,
                                activation=tf.nn.relu,
                                units=self.cfg["num_units"],
                                use_bias=True)
      self.head_rep = tf.layers.dropout(head_rep,
                                        rate=self.drop_rate,
                                        training=self.is_train)
      self.dep_rep = tf.layers.dropout(dep_rep,
                                       rate=self.drop_rate,
                                       training=self.is_train)
    print("head rep shape: {}".format(self.head_rep.get_shape().as_list()))
    print("dep  rep shape: {}".format(self.dep_rep.get_shape().as_list()))

  def _build_edge_dense_layer_op(self):
    with tf.variable_scope("edge_rep_dense"):
      self.edge_rep_dense = tf.get_variable(name="edge_rep_dense",
                                            shape=[self.cfg["num_units"],
                                                   self.cfg["num_units"]])

  def _build_edge_rep_op(self):
    with tf.variable_scope("edge_rep"):
      dep_rep = tf.expand_dims(self.dep_rep[:, 1:], axis=2)
      head_rep = tf.expand_dims(self.head_rep, axis=1)
      # 1D: batch_size, 2D: n_tokens (dep), 3D: n_tokens (head), 4D: dim
      edge_rep = self._create_edge_rep(dep_rep, head_rep)
      # 1D: batch_size, 2D: n_tokens (dep), 3D: n_tokens (head), 4D: dim
      self.edge_reps = tf.tensordot(edge_rep, self.edge_rep_dense,
                                    axes=[-1, -1])
      print("edge rep shape: {}".format(self.edge_reps.get_shape().as_list()))

  def _create_edge_rep(self, dep_rep, head_rep):
    if self.cfg["edge_rep"] == "minus":
      edge_rep = tf.subtract(dep_rep, head_rep)
    elif self.cfg["edge_rep"] == "multiply":
      edge_rep = tf.multiply(dep_rep, head_rep)
    else:
      edge_rep_s = tf.subtract(dep_rep, head_rep)
      edge_rep_m = tf.multiply(dep_rep, head_rep)
      edge_rep = edge_rep_s + edge_rep_m
    return edge_rep

  def _build_decoder_op(self):
    self._build_head_classifier_op()

  def _build_head_bilinear_classifier_op(self):
    with tf.variable_scope("head_classifier"):
      self.label_dense = tf.keras.layers.Dense(units=self.cfg["num_units"],
                                               use_bias=False)
      self.head_logits = tf.matmul(self.dep_rep[:, 1:],
                                   self.label_dense(self.head_rep),
                                   transpose_b=True)
      print("classifier: bilinear")

  def _build_head_classifier_op(self):
    with tf.variable_scope("head_classifier"):
      self.head_weights = tf.get_variable(name="head_weights",
                                          trainable=True,
                                          shape=self.cfg["num_units"])
      self.head_logits = self._compute_logits(self.edge_reps,
                                              self.head_weights)
      print("classifier: sim={}, scaling_factor={}".format(
        self.cfg["sim"], self.cfg["scaling_factor"]))

  def _compute_logits(self, reps_a, reps_b):
    if self.cfg["sim"] == "cos":
      logits = tf.tensordot(
        tf.nn.l2_normalize(reps_a, axis=-1),
        tf.nn.l2_normalize(reps_b, axis=-1),
        axes=[-1, -1]) * self.cfg["scaling_factor"]
    elif self.cfg["sim"] == "l2_eq_cos":
      logits = - (1 - tf.tensordot(
        tf.nn.l2_normalize(reps_a, axis=-1),
        tf.nn.l2_normalize(reps_b, axis=-1),
        axes=[-1, -1])) * self.cfg["scaling_factor"]
    else:
      logits = tf.tensordot(reps_a, reps_b, axes=[-1, -1])
    return logits

  def _build_loss_op(self):
    self._build_head_loss_op()
    self.loss = self.head_loss
    tf.summary.scalar("loss", self.loss)

  def _build_head_loss_op(self):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.head_logits, labels=self.heads)
    self.head_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=-1))
    tf.summary.scalar("head_loss", self.head_loss)

  def _build_train_op(self):
    with tf.variable_scope("train_step"):
      optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
      if self.cfg["grad_clip"] is not None and self.cfg["grad_clip"] > 0:
        grads, vs = zip(*optimizer.compute_gradients(self.loss))
        grads, _ = tf.clip_by_global_norm(grads, self.cfg["grad_clip"])
        self.train_op = optimizer.apply_gradients(zip(grads, vs))
      else:
        self.train_op = optimizer.minimize(self.loss)

  def _build_predict_op(self):
    self._build_head_prediction_op()

  def _build_head_prediction_op(self):
    self.head_predicts = tf.cast(
      tf.argmax(self.head_logits, axis=-1), tf.int32)
    self.head_proba = tf.nn.softmax(self.head_logits)

  def _build_answer_check_op(self):
    self._build_count_correct_heads_op()

  def _build_count_correct_heads_op(self):
    correct_heads = tf.cast(
      tf.equal(self.head_predicts, self.heads), tf.int32)
    n_tokens = (self.seq_len[0] - 1) * self.batch_size
    if self.cfg["include_puncts"]:
      self.correct_heads = correct_heads
      self.num_tokens = n_tokens
    else:
      not_puncts = 1 - self.puncts
      self.correct_heads = correct_heads * not_puncts
      self.num_tokens = n_tokens - tf.reduce_sum(self.puncts)
    self.num_correct_heads = tf.reduce_sum(self.correct_heads)

  def _add_bert_reps(self, batch, use_train_bert_hdf5):
    return self.batcher.load_and_add_bert_reps(batch, use_train_bert_hdf5)

  def train_epoch(self, batches):
    loss_total = 0.
    num_tokens = 0
    num_sents = 0
    num_batches = 0
    num_correct_heads = 0
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
                               self.num_correct_heads, self.num_tokens],
                              feed_dict)
      _, train_loss, num_cur_correct_heads, num_cur_tokens = outputs

      loss_total += train_loss
      num_correct_heads += num_cur_correct_heads
      num_tokens += num_cur_tokens
      num_sents += batch["batch_size"]

    avg_loss = loss_total / num_batches
    UAS = num_correct_heads / num_tokens
    seconds = time.time() - start_time
    self.logger.info("-- Train set")
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds)
    )
    self.logger.info("---- Loss: {:.2f} ({:.2f}/{:d})".format(
      avg_loss, loss_total, num_batches)
    )
    self.logger.info("---- UAS:{:>7.2%} ({:>6}/{:>6})".format(
      UAS, num_correct_heads, num_tokens)
    )
    return avg_loss, loss_total

  def evaluate_epoch(self, batches, data_name):
    num_tokens = 0
    num_sents = 0
    num_batches = 0
    num_correct_heads = 0
    start_time = time.time()

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      num_cur_correct_heads, num_cur_tokens = self.sess.run(
        [self.num_correct_heads, self.num_tokens], feed_dict)

      num_correct_heads += num_cur_correct_heads
      num_tokens += num_cur_tokens
      num_sents += batch["batch_size"]

    UAS = num_correct_heads / num_tokens
    seconds = time.time() - start_time
    self.logger.info("-- {} set".format(data_name))
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds)
    )
    self.logger.info("---- UAS:{:>7.2%} ({:>6}/{:>6})".format(
      UAS, num_correct_heads, num_tokens)
    )
    return UAS

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
    best_UAS = -np.inf
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
        self.cfg["lr"] = max(
          init_lr / (1.0 + self.cfg["lr_decay"] * epoch),
          self.cfg["minimal_lr"]
        )

      cur_valid_UAS = self.evaluate_epoch(valid_set, "Valid")

      if cur_valid_UAS > best_UAS:
        best_UAS = cur_valid_UAS
        self.save_session(epoch)
        self.logger.info(
          "-- new BEST UAS on Valid set: {:>7.2%}".format(best_UAS))

    self.train_writer.close()
    self.test_writer.close()

  def _preprocess_input_data(self, preprocessor):
    input_data = preprocessor.load_dataset(
      self.cfg["data_path"],
      keep_number=True,
      lowercase=self.cfg["char_lowercase"]
    )
    indexed_data = preprocessor.build_dataset(
      input_data[:self.cfg["data_size"]],
      self.word_dict,
      self.char_dict,
      self.pos_tag_dict,
      self.label_dict
    )
    write_json(os.path.join(self.cfg["save_path"], "tmp.json"), indexed_data)
    self.logger.info("Input sentences: {:>7}".format(len(indexed_data)))
    return input_data, indexed_data

  def make_one_batch(self, record):
    batch = defaultdict(list)
    for field in ["sent_id", "words", "chars", "pos_tags", "puncts"]:
      batch[field].append(record[field])
    return self.batcher.make_each_batch(batch)

  def eval(self, preprocessor):
    self.logger.info(str(self.cfg))
    raw_data = load_json(self.cfg["data_path"])[:self.cfg["data_size"]]
    _, indexed_data = self._preprocess_input_data(preprocessor)

    #############
    # Main loop #
    #############
    num_tokens = 0
    num_sents = len(indexed_data)
    num_correct_heads = 0
    start_time = time.time()

    print("PREDICTION START")
    for sent_id, (record, indexed_sent) in enumerate(zip(raw_data,
                                                         indexed_data)):
      assert record["sent_id"] == indexed_sent["sent_id"]
      if (sent_id + 1) % 100 == 0:
        print("%d" % (sent_id + 1), flush=True, end=" ")

      ##############
      # Prediction #
      ##############
      batch = self.make_one_batch(indexed_sent)
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      predicted_heads, head_proba = self.sess.run(
        [self.head_predicts, self.head_proba], feed_dict
      )
      record["predicted_heads"] = [int(head) for head in predicted_heads[0]]
      record["head_proba"] = [
        [float(Decimal(str(p)).quantize(
          Decimal("0.0001"), rounding=ROUND_HALF_UP))
         for p in proba] for proba in head_proba[0]]

      #########
      # Count #
      #########
      puncts = None if self.cfg["include_puncts"] else indexed_sent["puncts"]
      num_cur_correct_heads = self._count_correct_heads(
        gold_heads=record["heads"],
        predicted_heads=record["predicted_heads"],
        puncts=puncts
      )
      num_correct_heads += num_cur_correct_heads
      # -1 means excluding the ROOT node
      num_tokens += batch["seq_len"][0] - 1
      # - np.sum(batch["puncts"]) means excluding punctuations
      if self.cfg["include_puncts"] is False:
        num_tokens -= np.sum(batch["puncts"])

    accuracy = num_correct_heads / num_tokens
    seconds = time.time() - start_time
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds)
    )
    self.logger.info("---- Accuracy:{:>7.2%} ({:>6}/{:>6})".format(
      accuracy, num_correct_heads, num_tokens)
    )

    data_name = os.path.splitext(os.path.basename(self.cfg["data_path"]))[0]
    if self.cfg["output_file"]:
      if self.cfg["output_file"].endswith(".json"):
        file_name = self.cfg["output_file"]
      else:
        file_name = self.cfg["output_file"] + ".json"
      data_path = os.path.join(self.cfg["checkpoint_path"], file_name)
    else:
      data_path = os.path.join(self.cfg["checkpoint_path"],
                               data_name + ".predicted_heads.json")
    write_json(data_path, raw_data)

  def predict(self, preprocessor):
    self.logger.info(str(self.cfg))
    data_path = self.cfg["data_path"]
    if data_path.endswith(".json"):
      raw_data = load_json(data_path)[:self.cfg["data_size"]]
      _, indexed_data = self._preprocess_input_data(preprocessor)
    else:
      raw_data = preprocessor.load_txt_data(data_path)[:self.cfg["data_size"]]
      converted_data = preprocessor.convert_words_for_parsing(
        raw_data,
        keep_number=True,
        lowercase=self.cfg["char_lowercase"]
      )
      indexed_data = preprocessor.build_dataset(
        converted_data,
        self.word_dict,
        self.char_dict,
        self.pos_tag_dict,
        self.label_dict,
        add_heads=False,
        add_labels=False
      )
    write_json(os.path.join(self.cfg["save_path"], "tmp.json"), indexed_data)
    self.logger.info("Input sentences: {:>7}".format(len(indexed_data)))

    #############
    # Main loop #
    #############
    num_sents = len(indexed_data)
    start_time = time.time()

    print("PREDICTION START")
    for sent_id, (record, indexed_sent) in enumerate(zip(raw_data,
                                                         indexed_data)):
      if (sent_id + 1) % 100 == 0:
        print("%d" % (sent_id + 1), flush=True, end=" ")

      ##############
      # Prediction #
      ##############
      batch = self.make_one_batch(indexed_sent)
      if self.cfg["use_bert"]:
        batch = self._add_bert_reps(batch, use_train_bert_hdf5=False)
      feed_dict = self._get_feed_dict(batch)
      predicted_heads, head_proba = self.sess.run(
        [self.head_predicts, self.head_proba], feed_dict
      )
      record["predicted_heads"] = [int(head) for head in predicted_heads[0]]
      record["head_proba"] = [
        [float(Decimal(str(p)).quantize(
          Decimal("0.0001"), rounding=ROUND_HALF_UP))
         for p in proba] for proba in head_proba[0]]

    seconds = time.time() - start_time
    self.logger.info("---- Time: {:.2f} sec ({:.2f} sents/sec)".format(
      seconds, num_sents / seconds)
    )
    write_json(self.cfg["output_file"], raw_data)
