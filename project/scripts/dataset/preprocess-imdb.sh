#!/bin/bash


##################################################################
##  Constants                                                   ##
##################################################################

SOURCE_DIR="data/processed/raw/aclImdb"
TRAIN_DIR_NAME="train"
TEST_DIR_NAME="test"
DESTINATION_DIR="data/processed/ready-to-use/imdb"


##################################################################
##  Execution                                                   ##
##################################################################

python3 src/processing/preprocess_imdb.py -s ${SOURCE_DIR} \
                                          --train_dir_name ${TRAIN_DIR_NAME} \
                                          --test_dir_name ${TEST_DIR_NAME} \
                                          -d ${DESTINATION_DIR}
