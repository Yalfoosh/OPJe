#!/bin/bash


##################################################################
##  Constants                                                   ##
##################################################################

SOURCE_PATH="data/original/imdb.tar.gz"
DESTINATION_DIR="data/processed/raw"


##################################################################
##  Execution                                                   ##
##################################################################

python3 src/processing/unpack_imdb.py -s ${SOURCE_PATH} \
                                      -d ${DESTINATION_DIR}
