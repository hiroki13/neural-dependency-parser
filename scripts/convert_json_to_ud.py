# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import os
import ujson

WORD = 1
POS_TAG = 3
HEAD = 6
LABEL = 7


def load_conllu(filename):
  with codecs.open(filename, mode="r", encoding="utf-8") as f:
    words = []
    num_mwes = 0
    for line in f:
      line = line.lstrip().rstrip()
      if line.startswith("#"):
        continue
      if len(line) == 0:
        if len(words) != 0:
          yield words
          words = []
      else:
        line = line.split("\t")
        if line[0].isdigit():
          words.append(line)
        else:
          num_mwes += 1
  print("MWEs: %d" % num_mwes)


def load_json(filename):
  with codecs.open(filename, mode='r', encoding='utf-8') as f:
    dataset = ujson.load(f)
  return dataset


def main(argv):
  sents_conllu = list(load_conllu(argv.conllu_file))[:argv.data_size]
  sents_json = load_json(argv.json_file)[:argv.data_size]
  assert len(sents_conllu) == len(sents_json)

  os.makedirs(argv.output_dir, exist_ok=True)
  if argv.output_file:
    output_file = argv.output_file
  else:
    output_file = os.path.basename(argv.json_file)[:-5] + ".conllu"
  f = open(argv.output_dir + "/" + output_file, "w")
  is_pred_head = "predicted_heads" in sents_json[0]
  for sent_c, sent_j in zip(sents_conllu, sents_json):
    while True:
      if sent_c[0][0] == "#":
        line = sent_c.pop(0)
        f.write(line + "\n")
      else:
        break
    pred_heads = sent_j["predicted_heads"] if is_pred_head else sent_j["heads"]
    assert len(sent_c) == len(pred_heads)
    for line, head in zip(sent_c, pred_heads):
      line[HEAD] = str(int(head))
      f.write("\t".join(line) + "\n")
    f.write("\n")
  f.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='SCRIPT')
  parser.add_argument('--conllu_file',
                      help='path to conllu file')
  parser.add_argument('--json_file',
                      help='path to json file')
  parser.add_argument('--output_dir',
                      default="output_conllu",
                      help='output file name')
  parser.add_argument('--output_file',
                      default=None,
                      help='output file name')
  parser.add_argument('--data_size',
                      type=int,
                      default=100000000,
                      help='number of sentences to be used')
  main(parser.parse_args())
