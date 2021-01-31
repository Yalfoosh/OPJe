#!/bin/bash

##################################################################
##  Delete existing datasets                                    ##
##################################################################

echo "Deleting data/original/imdb.tar.gz..."
rm -rf ./data/original/imdb.tar.gz

echo "Deleting data/processed/raw/aclImdb..."
rm -rf ./data/processed/raw/aclImdb

echo "Deleting data/processed/ready-to-use/imdb..."
rm -rf ./data/processed/raw/ready-to-use/imdb

echo ""


##################################################################
##  Get punkt                                                   ##
##################################################################

python3 -c 'import nltk; nltk.download("punkt"); print()'


##################################################################
##  Download IMDB 50k dataset                                   ##
##################################################################

bash scripts/dataset/download-imdb.sh


##################################################################
##  Unpack it                                                   ##
##################################################################

bash scripts/dataset/unpack-imdb.sh


##################################################################
##  Preprocess it                                               ##
##################################################################

bash scripts/dataset/preprocess-imdb.sh
