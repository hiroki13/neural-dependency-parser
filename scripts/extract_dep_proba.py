import argparse
import codecs
import time
import ujson

ROOT_WORD = "_ROOT_"


def load_json(filename):
  with codecs.open(filename, mode='r', encoding='utf-8') as f:
    data = ujson.load(f)
  return data


def extract_with_gold_heads(data, only_errors=False):
  print("Preprocessing...")
  start_time = time.time()
  for sent in data:
    words = sent["words"]
    heads = sent["heads"]
    predicted_heads = sent["predicted_heads"]
    head_proba = sent["head_proba"]
    n_words = len(words)

    for i in range(n_words):
      dep_word = words[i]

      head = int(heads[i])
      head_word = ROOT_WORD if head == 0 else words[head - 1]

      p_head = int(predicted_heads[i])
      p_head_word = ROOT_WORD if p_head == 0 else words[p_head - 1]

      if only_errors and head == p_head:
        continue

      print("[%d %s] -> GOLD[%d %s] PRED[%d %s]" % (
        i + 1, dep_word, head, head_word, p_head, p_head_word)
      )
      for j, p in enumerate(head_proba[i]):
        if j == 0:
          print("-- %d\t%s\t%s" % (j, ROOT_WORD, str(p)))
        else:
          print("-- %d\t%s\t%s" % (j, words[j - 1], str(p)))
      print()
    print()

  print("Completed. Sent: {}, Time: {:.2f}".format(
    len(data), time.time() - start_time))


def extract(data):
  print("Preprocessing...")
  start_time = time.time()
  for sent in data:
    words = sent["words"]
    predicted_heads = sent["predicted_heads"]
    head_proba = sent["head_proba"]
    n_words = len(words)

    for i in range(n_words):
      dep_word = words[i]

      p_head = int(predicted_heads[i])
      p_head_word = ROOT_WORD if p_head == 0 else words[p_head - 1]

      print("[%d %s] -> PRED[%d %s]" % (
        i + 1, dep_word, p_head, p_head_word)
      )
      for j, p in enumerate(head_proba[i]):
        if j == 0:
          print("-- %d\t%s\t%s" % (j, ROOT_WORD, str(p)))
        else:
          print("-- %d\t%s\t%s" % (j, words[j - 1], str(p)))
      print()
    print()

  print("Completed. Sent: {}, Time: {:.2f}".format(
    len(data), time.time() - start_time))


def main(args):
  data_json = load_json(args.json)[:args.data_size]
  if args.gold_heads:
    extract_with_gold_heads(data_json, args.only_errors)
  else:
    extract(data_json)


if __name__ == '__main__':
  if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCRIPT')
    parser.add_argument('--json',
                        help='path to json file')
    parser.add_argument('--data_size',
                        default=1000000000,
                        type=int,
                        help='Data size')
    parser.add_argument('--gold_heads',
                        action='store_true',
                        default=False,
                        help='whether or not to extract gold heads')
    parser.add_argument('--only_errors',
                        action='store_true',
                        default=False,
                        help='whether or not to print only errors')
    main(parser.parse_args())
