# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import json

from models.labeled_models import LabeledWeightBasedModel, \
  LabeledInstanceBasedModel
from models.multitask_models import MultiTaskWeightBasedModel, \
  MultiTaskInstanceBasedModel
from utils.batchers import WeightBatcher, LabeledInstanceBatcher, \
  BERTWeightBatcher, BERTLabeledInstanceBatcher
from utils.preprocessors import Preprocessor


def set_config(args, config):
  if args.raw_path:
    config["raw_path"] = args.raw_path
  if args.save_path:
    config["save_path"] = args.save_path
  if args.data_path:
    config["data_path"] = args.data_path
  if args.bert_hdf5:
    config["valid_bert_hdf5"] = args.bert_hdf5
  if args.checkpoint_path:
    config["checkpoint_path"] = args.checkpoint_path
    config["summary_path"] = os.path.join(args.checkpoint_path, "summary")
  if args.summary_path:
    config["summary_path"] = args.summary_path
  if args.task_name:
    config["task_name"] = args.task_name
  if args.model_name:
    config["model_name"] = args.model_name
  if args.batch_size:
    config["batch_size"] = args.batch_size
  if args.data_size:
    config["data_size"] = args.data_size
  if args.classifier:
    config["classifier"] = args.classifier
  if args.sim:
    config["sim"] = args.sim
  if args.decode:
    config["decode"] = args.decode
  if args.k:
    config["k"] = args.k
  if args.unuse_all_instances:
    config["use_all_instances"] = False
  config["use_gold_heads"] = args.use_gold_heads
  config["use_nearest_sents"] = args.use_nearest_sents
  config["output_file"] = args.output_file
  return config


def main(args):
  config = json.load(open(args.config_file))
  config = set_config(args, config)

  preprocessor = Preprocessor(config)

  if config["decode"] == "weight":
    if config["use_bert"]:
      batcher = BERTWeightBatcher(config)
    else:
      batcher = WeightBatcher(config)
    if config["task_name"] == "multi":
      model = MultiTaskWeightBasedModel(config, batcher)
    else:
      model = LabeledWeightBasedModel(config, batcher)
  else:
    if config["use_bert"]:
      batcher = BERTLabeledInstanceBatcher(config)
    else:
      batcher = LabeledInstanceBatcher(config)
    if config["task_name"] == "multi":
      model = MultiTaskInstanceBasedModel(config, batcher)
    else:
      model = LabeledInstanceBasedModel(config, batcher)

  model.restore_last_session(config["checkpoint_path"])

  if args.mode == "eval":
    model.eval(preprocessor)
  else:
    model.save_edge_representation(preprocessor)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode',
                      default='eval',
                      help='eval/edge_rep')
  parser.add_argument('--config_file',
                      required=True,
                      default='data/config/config.json',
                      help='Configuration file')
  parser.add_argument('--data_name',
                      default='valid',
                      help='Data name')
  parser.add_argument('--data_path',
                      default='data/ptb/valid.json',
                      help='Path to data')
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
  parser.add_argument('--task_name',
                      default=None,
                      help='Task name')
  parser.add_argument('--model_name',
                      default=None,
                      help='Model name')
  parser.add_argument('--batch_size',
                      default=None,
                      type=int,
                      help='Batch size')
  parser.add_argument('--data_size',
                      default=None,
                      type=int,
                      help='Data size')
  parser.add_argument('--classifier',
                      default=None,
                      help='cos/dot')
  parser.add_argument('--sim',
                      default=None,
                      help='cos/dot')
  parser.add_argument('--decode',
                      default=None,
                      help='classifier-based/instance-based')
  parser.add_argument('--k',
                      default=None,
                      type=int,
                      help='k nearest sentences')
  parser.add_argument('--use_gold_heads',
                      action='store_true',
                      default=False,
                      help='whether or not to use predicted heads')
  parser.add_argument('--use_nearest_sents',
                      action='store_true',
                      default=False,
                      help='whether or not to use nearest sentences')
  parser.add_argument('--unuse_all_instances',
                      action='store_true',
                      default=False,
                      help='whether or not to use all training instances')
  parser.add_argument('--bert_hdf5',
                      default='data/ptb/valid.subwords.first_bert_reps.hdf5',
                      help='Path to bert hdf5 file')
  parser.add_argument('--output_file',
                      default=None,
                      help='output file name')
  main(parser.parse_args())
