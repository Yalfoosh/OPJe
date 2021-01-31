#!/bin/bash


##################################################################
##  Constants                                                   ##
##################################################################

DOWNLOAD_URL="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DESTINATION_DIR="data/original/imdb.tar.gz"
N_RETRIES=3


##################################################################
##  Execution                                                   ##
##################################################################

wget ${DOWNLOAD_URL} -O ${DESTINATION_DIR} \
                     -t ${N_RETRIES} \
                     -q \
                     --show-progress
