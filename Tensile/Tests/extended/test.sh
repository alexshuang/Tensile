#!/bin/sh

bin=../../bin/Tensile
TEST_DIR="tensor_contraction/"
out=out
legacy=${legacy:-0}
if [ $legacy -eq 0 ]; then
    res_dir=res_dir
else
    res_dir=res_dir/legacy
fi
SUMMARY=$res_dir/summary

rm -rf $res_dir
mkdir -p $res_dir

for o in $TEST_DIR; do
    YAMLS=`find $o -name \*.yaml`
    for y in $YAMLS; do
        FILENAME=${y##*/}
        BASENAME=${FILENAME%.*}
        sed -i '/.*ThreadTile/,/.*WorkGroup/s/\(.*- \[\).*\]/\11, 32]/g' $y
        sed -i '/- KernelLanguage/d' $y
        sed -i '/\<ForkParameters:/a\ \ \ \ - KernelLanguage: ["Assembly"]' $y
        if [ $legacy -eq 0 ]; then
            sed -i '/- MatrixInstruction/d' $y
            sed -i '/\<ForkParameters:/a\ \ \ \ - MatrixInstruction: [[32, 32, 1, 2]]' $y
        fi
        $bin $y $out | tee $res_dir/$BASENAME

        lines=`wc -l $res_dir/$BASENAME | awk '{print $1}'`
        if [ $lines -lt 20 ]; then
            sed -i '/- KernelLanguage/d' $y
            sed -i '/\<ForkParameters:/a\ \ \ \ \ \ \ \ - KernelLanguage: ["Assembly"]' $y
            if [ $legacy -eq 0 ]; then
                sed -i '/- MatrixInstruction/d' $y
                sed -i '/\<ForkParameters:/a\ \ \ \ \ \ \ \ - MatrixInstruction: [[32, 32, 1, 2]]' $y
            fi
            $bin $y $out | tee $res_dir/$BASENAME
        fi
        echo $BASENAME >> $SUMMARY
        tail -3 $res_dir/$BASENAME >> $SUMMARY
    done
done
