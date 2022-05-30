# Neural Dependency Parser

## Prerequisites
* [python3](https://www.python.org/downloads/)
* [TensorFlow](https://www.tensorflow.org/)
* [h5py](https://www.h5py.org/)

## Installation
- CPU
```
conda create -n dep-tf python=3.6
source activate dep-tf
conda install -c conda-forge tensorflow
pip install ujson tqdm
git clone https://github.com/hiroki13/neural-dependency-parser.git
```
- GPU
```
conda create -n dep-tf python=3.6
source activate dep-tf
pip install tensorflow-gpu==1.10 ujson tqdm
git clone https://github.com/hiroki13/neural-dependency-parser.git
```

## Get Started
### Data Preparation
- Download datasets and convert them to json files:
`./create_ud_datasets.sh`
- Download a [Japanese pre-trained model](https://drive.google.com/file/d/1mPW-0qPGPEYVl3TgSPH6OQUsWVpx3fnM/view?usp=sharing)
### Run a parser with the downloaded model:
```
python run_unlabeled_models.py --mode pred --config_file ckpt_ja_gsd-ud_unlabeled_weight_keep08_0/config.json --data_path data/example/example.ja.txt --output_file heads_predicted.json
```
- You will obtain the output file `heads_predicted.json`
### Print dependencies and head probabilities produced by the trained parser:
```
python scripts/extract_dep_proba.py --json heads_predicted.json
```
- The following lines will be printed:
```
[1 夏目] -> PRED[2 漱石]
-- 0    _ROOT_  0.0
-- 1    夏目    0.0
-- 2    漱石    1.0
-- 3    は      0.0
-- 4    、日本  0.0
-- 5    の      0.0

...

-- 15   文豪    0.0
-- 16   の      0.0
-- 17   一人    1.0
-- 18   。      0.0
```

## Training Your Own Parsers
### Embedding Preparation
- Go to `https://fasttext.cc/docs/en/crawl-vectors.html` and download the embeddings you want
- Or download as follows:
```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
gunzip　cc.ja.300.vec.gz
```
- Put the embedding file to `data/emb`
```
mkdir data/emb
mv cc.ja.300.vec data/emb
```

### Train a parser
- To train a parser (model), run the following line:
```
python train_unlabeled_models.py --config_file data/config/config.weight.ud.json
```
- The trained parser (model) will be saved at `checkpoint_ja_gsd-ud_fasttext`

### Run the trained parser
- To use the trained parser for new sentences, run the following line:
```
python run_unlabeled_models.py --mode eval --config_file checkpoint_ja_gsd-ud_fasttext/config.json --data_path data/ja_gsd-ud/valid.json
```

### Print dependencies and head probabilities produced by the trained parser
```
python scripts/extract_dep_proba.py --json checkpoint_ja_gsd-ud_fasttext/valid.predicted_heads.json
```

## LICENSE
MIT License