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
git clone https://github.com/hiroki13/neural-dep-parser.git
```
- GPU
```
conda create -n dep-tf python=3.6
source activate dep-tf
pip install tensorflow-gpu==1.10 ujson tqdm
git clone https://github.com/hiroki13/neural-dep-parser.git
```

## Data Preparation
`./create_ud_datasets.sh`

## Embedding Preparation
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

## Training
```
python train_unlabeled_models.py --config_file data/config/config.weight.ud.json
```

## Run the trained parser
```
python run_unlabeled_models.py --config_file checkpoint_ja_gsd-ud_fasttext/config.json --data_path data/ja_gsd-ud/valid.json
```

## Extract head probability produced by the trained parser
```
python scripts/extract_dep_proba.py --json checkpoint_ja_gsd-ud_fasttext/valid.predicted_heads.json
```