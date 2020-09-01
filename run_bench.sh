#!/bin/sh

OUT_DIR=${1:-out}
YAMLS=`find Tensile/Configs/ -name *.yaml -maxdepth 1 | grep -E '_hgemm_' | grep -v '_lite' | grep -v '_single_' | grep -v '_bufferload_'`

#echo $YAMLS
#exit 0

istart=$(date +%s)

set -x

for o in $YAMLS; do
	FILENAME=${o##*/}
	BASENAME=${FILENAME%%.*}
	OUT=$OUT_DIR/$BASENAME
	mkdir -p $OUT
	cp $o $OUT/config.yaml -v
	Tensile/bin/Tensile $o $OUT | tee $OUT/run.log
done

iend=$(date +%s)

echo "Benchmark Done."
echo "The total time: $(( iend - istart )) seconds."
