# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import json

from models.unlabeled_models import UnlabeledWeightBasedModel
from utils.batchers import WeightBatcher, BERTWeightBatcher
from utils.preprocessors import Preprocessor
from utils.common import get_vocab_size_and_dim


def set_config(args, config):
  if args.raw_path:
    config["raw_path"] = args.raw_path
  if args.save_path:
    config["save_path"] = args.save_path
    config["train_set"] = os.path.join(args.save_path, "train.json")
    config["valid_set"] = os.path.join(args.save_path, "valid.json")
    config["vocab"] = os.path.join(args.save_path, "vocab.json")
    config["pretrained_emb"] = os.path.join(args.save_path, "emb.npz")
  if args.epochs:
    config["epochs"] = args.epochs
  if args.train_set:
    config["train_set"] = args.train_set
  if args.valid_set:
    config["valid_set"] = args.valid_set
  if args.pretrained_emb:
    config["pretrained_emb"] = args.pretrained_emb
  if args.vocab:
    config["vocab"] = args.vocab
  if args.checkpoint_path:
    config["checkpoint_path"] = args.checkpoint_path
    config["summary_path"] = os.path.join(args.checkpoint_path, "summary")
  if args.summary_path:
    config["summary_path"] = args.summary_path
  if args.emb_path:
    config["emb_path"] = args.emb_path
  if args.model_name:
    config["model_name"] = args.model_name
  if args.batch_size:
    config["batch_size"] = args.batch_size
  if args.data_size:
    config["data_size"] = args.data_size
  if args.edge_rep:
    config["edge_rep"] = args.edge_rep
  if args.sim:
    config["sim"] = args.sim
  if args.vocab_size:
    config["vocab_size"] = args.vocab_size
  if args.emb_dim:
    config["emb_dim"] = args.emb_dim
  if args.num_units:
    config["num_units"] = args.num_units
  if args.num_layers:
    config["num_layers"] = args.num_layers
  if args.keep_prob:
    config["keep_prob"] = args.keep_prob
  if args.k:
    config["k"] = args.k
  if args.num_negatives:
    config["num_negatives"] = args.num_negatives
  if args.scaling_factor:
    config["scaling_factor"] = args.scaling_factor
  if args.train_bert_hdf5:
    config["train_bert_hdf5"] = args.train_bert_hdf5
    config["use_bert"] = True
  if args.valid_bert_hdf5:
    config["valid_bert_hdf5"] = args.valid_bert_hdf5
    config["use_bert"] = True
  if args.bert_keep_prob:
    config["bert_keep_prob"] = args.bert_keep_prob
  if args.unuse_words:
    config["use_words"] = False
  if args.unuse_chars:
    config["use_chars"] = False
  if args.unuse_pos_tags:
    config["use_pos_tags"] = False
  config["include_puncts"] = args.include_puncts
  return config


def main(args):
  config = json.load(open(args.config_file))
  config = set_config(args, config)
  preprocessor = Preprocessor(config)

  # create dataset from raw data files
  if not os.path.exists(config["save_path"]):
    preprocessor.preprocess()
  config["vocab_size"], config["emb_dim"] = get_vocab_size_and_dim(
    config["pretrained_emb"])

  if config["use_bert"]:
    batcher = BERTWeightBatcher(config)
  else:
    batcher = WeightBatcher(config)
  model = UnlabeledWeightBasedModel(config, batcher)
  model.train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file',
                      required=True,
                      default='data/config/config.json',
                      help='Configuration file')
  parser.add_argument('--raw_path',
                      default=None,
                      help='Raw data directory')
  parser.add_argument('--save_path',
                      default=None,
                      help='Save directory')
  parser.add_argument('--checkpoint_path',
                      default=None,
                      help='Checkpoint directory')
  parser.add_argument('--summary_path',
                      default=None,
                      help='Summary directory')
  parser.add_argument('--emb_path',
                      default=None,
                      help='Embedding directory')
  parser.add_argument('--model_name',
                      default=None,
                      help='Model name')
  parser.add_argument('--epochs',
                      default=None,
                      type=int,
                      help='number of epochs')
  parser.add_argument('--batch_size',
                      default=None,
                      type=int,
                      help='Batch size')
  parser.add_argument('--train_set',
                      default=None,
                      help='path to training set')
  parser.add_argument('--valid_set',
                      default=None,
                      help='path to training set')
  parser.add_argument('--pretrained_emb',
                      default=None,
                      help='path to pretrained embeddings')
  parser.add_argument('--vocab',
                      default=None,
                      help='path to vocabulary')
  parser.add_argument('--data_size',
                      default=None,
                      type=int,
                      help='Data size')
  parser.add_argument('--edge_rep',
                      default=None,
                      help='minus/multiply/minus_and_multiply')
  parser.add_argument('--sim',
                      default=None,
                      help='cos/dot')
  parser.add_argument('--vocab_size',
                      default=None,
                      type=int,
                      help='number of vocab size')
  parser.add_argument('--emb_dim',
                      default=None,
                      type=int,
                      help='number of embedding dimensions')
  parser.add_argument('--num_units',
                      default=None,
                      type=int,
                      help='number of RNN hidden units')
  parser.add_argument('--num_layers',
                      default=None,
                      type=int,
                      help='number of RNN layers')
  parser.add_argument('--keep_prob',
                      default=None,
                      type=float,
                      help='Keep (dropout) probability')
  parser.add_argument('--k',
                      default=None,
                      type=int,
                      help='k-NN sentences')
  parser.add_argument('--num_negatives',
                      default=None,
                      type=int,
                      help='number of negative samples')
  parser.add_argument('--scaling_factor',
                      default=None,
                      type=float,
                      help='scaling factor for arc face')
  parser.add_argument('--train_bert_hdf5',
                      default=None,
                      type=str,
                      help='path to train bert hdf5')
  parser.add_argument('--valid_bert_hdf5',
                      default=None,
                      type=str,
                      help='path to valid bert hdf5')
  parser.add_argument('--bert_keep_prob',
                      default=None,
                      type=float,
                      help='BERT Keep (dropout) probability')
  parser.add_argument('--unuse_words',
                      action='store_true',
                      default=False,
                      help='whether or not to use word embeddings')
  parser.add_argument('--unuse_chars',
                      action='store_true',
                      default=False,
                      help='whether or not to use character embeddings')
  parser.add_argument('--unuse_pos_tags',
                      action='store_true',
                      default=False,
                      help='whether or not to use pos tag embeddings')
  parser.add_argument('--include_puncts',
                      action='store_true',
                      default=False,
                      help='whether or not to include punctuations')
  main(parser.parse_args())
