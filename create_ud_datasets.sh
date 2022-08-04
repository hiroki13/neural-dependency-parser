#! /bin/bash

# (Arabic-PADT, ar_padt-ud), (Basque-BDT, eu_bdt-ud), (Chinese-GSD, zh_gsd-ud),
# (English-EWT, en_ewt-ud), (Finnish-TDT, fi_tdt-ud), (Hebrew-HTB, he_htb-ud),
# (Hindi-HDTB, hi_hdtb-ud), (Italian-ISDT, it_isdt-ud), (Japanese-GSD, ja_gsd-ud),
# (Korean-GSD, ko_gsd-ud), (Russian-SynTagRus, ru_syntagrus-ud), (Swedish-Talbanken, sv_talbanken-ud),
# (Turkish-IMST, tr_imst-ud)

# UD download page: https://universaldependencies.org/#download
# Download UD datasets (download the latest version)
#wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz
#tar -xzvf ud-treebanks-v2.7.tgz
#mv ud-treebanks-v2.7 data/

# Select one file from data/ud-treebanks-v2.7
dir_name=UD_Japanese-GSD
# Make a new directory for the dataset to convert
mv data/ud-treebanks-v2.7/$dir_name data
data_name=ja_gsd-ud
# Convert a conllu file to its json file
python scripts/convert_ud_to_json.py --input_file data/$dir_name/"$data_name"-train.conllu --output_file data/$dir_name/train.json
python scripts/convert_ud_to_json.py --input_file data/$dir_name/"$data_name"-dev.conllu --output_file data/$dir_name/valid.json
python scripts/convert_ud_to_json.py --input_file data/$dir_name/"$data_name"-test.conllu --output_file data/$dir_name/test.json
