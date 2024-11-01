#!/bin/bash

RUNNING_FILENAME=$1
DATA_FILENAME=$2
SAMPLE_QUERY_FILENAME=$3
QUERY_FILENAME=$4

g++ -std=gnu++17 -Wall -Wextra -Ofast -march=native -o a.out "$RUNNING_FILENAME"

./a.out "$DATA_FILENAME" "$SAMPLE_QUERY_FILENAME" "$QUERY_FILENAME"

rm ./a.out