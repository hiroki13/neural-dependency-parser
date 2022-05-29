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

## Data Preparation
- Download datasets and convert them to json files:
`./create_ud_datasets.sh`
- Download a [Japanese pre-trained model](https://drive.google.com/file/d/1mPW-0qPGPEYVl3TgSPH6OQUsWVpx3fnM/view?usp=sharing)

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

## Training a parser
- To train a parser (model), run the following line:
```
python train_unlabeled_models.py --config_file data/config/config.weight.ud.json
```
- The trained parser (model) will be saved at `checkpoint_ja_gsd-ud_fasttext`

## Run the trained parser
- To use the trained parser for new sentences, run the following line:
```
python run_unlabeled_models.py --config_file checkpoint_ja_gsd-ud_fasttext/config.json --data_path data/ja_gsd-ud/valid.json
```

## Extract head probability produced by the trained parser
```
python scripts/extract_dep_proba.py --json checkpoint_ja_gsd-ud_fasttext/valid.predicted_heads.json
```

## LICENSE
MIT License