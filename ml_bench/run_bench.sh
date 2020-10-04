#!/bin/sh

TOP_DIR=$PWD/../

CONF_DIR=${1:-../Tensile/Configs}
OUT_DIR=${2:-out}

YAMLS=`find $CONF_DIR -name *.yaml -maxdepth 1 | grep -E '_hgemm_' | grep -v '_lite' | grep -v '_single_' | grep -v '_bufferload_'`

istart=$(date +%s)

set -ex

for o in $YAMLS; do
	FILENAME=${o##*/}
	BASENAME=${FILENAME%%.*}
	TENSILE_OUT=$OUT_DIR/$BASENAME
    BENCH_OUT=$OUT_DIR/bench_results/$BASENAME
	mkdir -p $TENSILE_OUT $BENCH_OUT
	cp $o $TENSILE_OUT/config.yaml -v
	$TOP_DIR/Tensile/bin/Tensile $o $TENSILE_OUT | tee $TENSILE_OUT/run.log
    cp $TENSILE_OUT/2_BenchmarkData/* $BENCH_OUT -av
done

iend=$(date +%s)

echo "Benchmark Done."
echo "The total time: $(( iend - istart )) seconds."
