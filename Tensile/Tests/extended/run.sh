#!/bin/sh

bin=../../bin/Tensile
TEST_DIR=$1
out=out
legacy=${legacy:-0}
if [ $legacy -eq 0 ]; then
    res_dir=res_dir
else
    res_dir=res_dir/legacy
fi
SUMMARY=$res_dir/summary

if [ "${TEST_DIR}" -eq "" ]; then
    echo "run.sh <path/to/test cases>"
    exit 2
fi

rm -rf $res_dir
mkdir -p $res_dir
cnt=0
for o in $TEST_DIR; do
    YAMLS=`find $o -name \*.yaml`
    for y in $YAMLS; do
        FILENAME=${y##*/}
        BASENAME=${FILENAME%.*}
        sed -i '/.*ThreadTile/,/.*WorkGroup/s/\(.*- \[\).*\]/\11, 32]/g' $y
        sed -i '/- KernelLanguage/d' $y
        sed -i '/- MatrixInstruction/d' $y
        sed -i 's/\(\s*\)- DepthU:\(.*\)/\1- DepthU:\2\n\1- KernelLanguage: ["Assembly"]\n\1- MatrixInstruction: [[32, 32, 1, 2]]/g' $y
        $bin $y $out | tee $res_dir/$BASENAME
        echo "[$cnt] $FILENAME:" >> $SUMMARY
        tail -5 $res_dir/$BASENAME >> $SUMMARY
        echo "" >> $SUMMARY
        cnt=$(($cnt+1))
    done
done
