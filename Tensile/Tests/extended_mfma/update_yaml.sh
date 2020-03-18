#!/bin/sh

TEST_DIR=$1

YAMLS=`find $TEST_DIR -name \*.yaml`
for y in $YAMLS; do
#    sed -i '/^\s*- ThreadTile/,/^\s*- WorkGroup:\n\s*- \[\s*\d{1,2}\s*,\s*\d{1,2}\s*,\s*1\s*]/d' $y
        sed -i '/.*ThreadTile/,/.*WorkGroup/s/\(^\s*\)- \(\[.*\]\)/\1- [1, 32]/g' $y
        sed -i '/.*- WorkGroup/,/.*- DepthU/s/\(^\s*\)- \(\[.*\]\)/\1- [16, 16, 1]\n\1- [64, 4, 1]/g' $y
    sed -i '/- KernelLanguage/d' $y
    sed -i '/- MatrixInstruction/d' $y
#    sed -i 's/\(\s*\)- DepthU:\(.*\)/\1- DepthU:\2\n\1- KernelLanguage: [\"Assembly\"]\n\1- MatrixInstruction: [[32, 32, 1, 2]]\n\1- ThreadTile:\n\1  - [1, 32]\n\1- WorkGroup:\n\1  - [16, 16, 1]\n\1  - [64, 4, 1]/g' $y
    sed -i 's/\(\s*\)- DepthU:\(.*\)/\1- DepthU:\2\n\1- KernelLanguage: [\"Assembly\"]\n\1- MatrixInstruction: [[32, 32, 1, 2]]/g' $y
done
