#!/bin/bash


##################################################################
##  Constants                                                   ##
##################################################################

TRAINING_SET_PATH="data/processed/ready-to-use/imdb/train.tsv"
TEST_SET_PATH="data/processed/ready-to-use/imdb/test.tsv"
MODEL_PATH="models/prado-imdb"

FEATURE_LENGTH=128
EMBEDDING_LENGTH=64
DROPOUT=0.2
OUT_CHANNELS=16
SKIPGRAM_PATTERNS=(\
    "1" \
    "11" \
    "111"   "101" \
    "1111"  "1101"  "1001" \
    "11111" "11101" "11011" "10101" "10001" \
)
OUT_FEATURES=2

DEVICE="cpu"
N_EPOCHS=100
LEARNING_RATE=3e-4
BATCH_SIZE=16
EVALUATION_FREQUENCY=1
INSERTION_PROBABILITY=0.01
DELETION_PROBABILITY=0.01
SWAP_PROBABILITY=0.01


##################################################################
##  Execution                                                   ##
##################################################################

python3 src/train.py \
    --training_set_path ${TRAINING_SET_PATH} \
    --test_set_path ${TEST_SET_PATH} \
    --model_path ${MODEL_PATH} \
    --feature_length ${FEATURE_LENGTH} \
    --embedding_length ${EMBEDDING_LENGTH} \
    --dropout ${DROPOUT} \
    --out_channels ${OUT_CHANNELS} \
    --skipgram_patterns $(echo ${SKIPGRAM_PATTERNS[@]}) \
    --out_features ${OUT_FEATURES} \
    --device ${DEVICE} \
    --n_epochs ${N_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --insertion_probability ${INSERTION_PROBABILITY} \
    --deletion_probability ${DELETION_PROBABILITY} \
    --swap_probability ${SWAP_PROBABILITY} \
    --evaluation_frequency ${EVALUATION_FREQUENCY} \
    --autoresume
