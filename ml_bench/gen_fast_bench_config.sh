#!/bin/sh

SRC=$1
FILENAME=${SRC##*/}
PARENT=${SRC%/*}
OUTPUT=${PARENT}/fast_bench_${FILENAME}

cp $SRC $OUTPUT -v
sed -i '/^GlobalParameters:/a\  FastBenchmark: True\n  FastSolutionKeep: 0.15' $OUTPUT
