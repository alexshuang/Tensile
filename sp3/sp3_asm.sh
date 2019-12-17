#!/bin/bash

set -e

TOP_DIR=${RKT_DIR:-$PWD}
BIN_DIR=$TOP_DIR/bin
OUT_DIR=$TOP_DIR/out
export LD_LIBRARY_PATH=$BIN_DIR

if [ "$#" -eq "0" ]; then
	echo "Usage: sp3_asm.sh <source file|source dir>"
	exit -1
fi

if [ -d $1 ]; then
	SRC=`find $1 -name *.sp3`
else
	SRC=$1
fi

mkdir -p $OUT_DIR

for o in $SRC; do
	BASENAME=${o%.*}
	BASENAME=$OUT_DIR/${BASENAME##*/}
	HEX=$BASENAME.hex
	OUT=${BASENAME}_out.sp3
	INC=$BASENAME.inc
	$BIN_DIR/mi100_sp3 $o asic=MI9 type=cs -hex $HEX
	$BIN_DIR/mi100_sp3 -hex $HEX asic=MI9 type=cs  $OUT
	cat $OUT | grep "// " | grep ": " | sed "s/.*: //" | sed "s/ \([0-9a-f]*\)/, 0x\1/" |sed "s/^/.long 0x/" > $INC
	#/opt/rocm/opencl/bin/x86_64/clang -x assembler -target amdgcn-amdhsa -mcpu=gfx906 template.s -o output.co
	echo
	echo
	echo "[Finish]: $o -> $INC"
	echo
	echo
done

