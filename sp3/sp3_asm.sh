#!/bin/bash

set -e

export LD_LIBRARY_PATH=$PWD

if [ "$#" -eq "0" ]; then
	echo "Usage: sp3_asm.sh <source file|source dir>"
	exit -1
fi

if [ -d $1 ]; then
	SRC=`find $1 -name *.sp3`
else
	SRC=$1
fi

for o in $SRC; do
	HEX=${o%.*}.hex
	OUT=${o%.*}_out.sp3
	INC=${o%.*}.inc
	$PWD/mi100_sp3 $o asic=MI9 type=cs -hex $HEX
	$PWD/mi100_sp3 -hex $HEX asic=MI9 type=cs  $OUT
	cat $OUT | grep "// " | grep ": " | sed "s/.*: //" | sed "s/ \([0-9a-f]*\)/, 0x\1/" |sed "s/^/.long 0x/" > $INC
	#/opt/rocm/opencl/bin/x86_64/clang -x assembler -target amdgcn-amdhsa -mcpu=gfx906 template.s -o output.co
	echo
	echo
	echo "[Finish]: $o -> $INC"
	echo
	echo
done

