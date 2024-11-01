#!/bin/bash

DIMENSION=$1
NUM_DATA=$2
NUM_SAMPLE_QUERY=$3
NUM_QUERY=$4
DATA_FILENAME=$5
SAMPLE_QUERY_FILENAME=$6
QUERY_FILENAME=$7

g++ -std=gnu++17 -Wall -Wextra -O2 -o ./data/generator ./data/generator.cpp

./data/generator "$DIMENSION" "$NUM_DATA" "$NUM_SAMPLE_QUERY" "$NUM_QUERY" "$DATA_FILENAME" "$SAMPLE_QUERY_FILENAME" "$QUERY_FILENAME"

rm ./data/generator