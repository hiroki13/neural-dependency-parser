# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import ujson
import unicodedata

WORD = 1
POS_TAG = 3
HEAD = 6
LABEL = 7


def load(filename):
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
          words.append(
            (line[WORD], line[POS_TAG], int(line[HEAD]), line[LABEL]))
        else:
          num_mwes += 1
  print("MWEs: %d" % num_mwes)


def write_json(filename, data):
  with codecs.open(filename, mode="w", encoding="utf-8") as f:
    ujson.dump(data, f, ensure_ascii=False)


def is_punct(word):
  return all(unicodedata.category(char).startswith('P') for char in word)


def main(argv):
  sents = list(load(argv.input_file))
  data = []
  n_sents = 0
  n_words = 0
  n_puncts = 0
  for sent in sents:
    words, pos_tags, heads, labels = list(zip(*sent))
    data.append({"sent_id": n_sents,
                 "words": words,
                 "pos_tags": pos_tags,
                 "heads": heads,
                 "labels": labels})
    n_sents += 1
    n_words += len(words)
    for word in words:
      if is_punct(word):
        n_puncts += 1

  if argv.output_file.endswith(".json"):
    path = argv.output_file
  else:
    path = argv.output_file + ".json"
  write_json(path, data)
  print("Sents:%d Words:%d (Punctuations:%d)" % (n_sents, n_words, n_puncts))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='SCRIPT')
  parser.add_argument('--input_file',
                      help='path to ud file')
  parser.add_argument('--output_file',
                      default="output.json",
                      help='output file name')
  main(parser.parse_args())
