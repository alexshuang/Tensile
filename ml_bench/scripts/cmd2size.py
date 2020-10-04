#!/usr/bin/env python3
import os
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "CMD of rocblas-bench -> Problem size"
    parser.add_argument("-f", "--config", type=str, help='config file path')
    parser.add_argument("-o", "--out_dir", type=str, default='out', help='output dir')
    args = parser.parse_args()

    if not args.config:
        print('No problem sizes specified, please use "--help" for more details.')
        exit(-1)

    src_path = Path(args.config)
    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    files = list(src_path) if src_path.is_file() else src_path.glob("**/rocblas_bench.csv")
    nn, nt, tn, tt = [], [], [], []

    for f in files:
        sizes = open(f).readlines()

        #import pdb; pdb.set_trace()

        for o in sizes:
            s = o.split(' ')
            if s[2] == 'gemm_strided_batched_ex':
                size = "- Exact: { sizes: [%s, %s, %s, %s], stridesA: [-1, %s, %s], stridesB: [-1, %s, %s], stridesC: [-1, %s, %s], stridesD: [-1, %s, %s] }\n" % (s[8], s[10], s[42], s[12], s[18], s[20], s[24], s[26], s[32], s[34], s[38], s[40])
            else:
                size = "- Exact: [{}]\n".format(', '.join([s[8], s[10], '1', s[12], s[28], s[32], s[18], s[22]]))
            transA, transB = s[4], s[6]
            if transA == 'N' and transB == 'N':
                nn.append(size)
            elif transA == 'N' and transB == 'T':
                nt.append(size)
            elif transA == 'T' and transB == 'N':
                tn.append(size)
            elif transA == 'T' and transB == 'T':
                tt.append(size)
            else:
                print("unkown transA: {} and transB: {}".format(transA, transB))

    if len(nn):
        with open(out_path/'nn_sizes.yaml', 'w+') as fp:
            for o in set(nn):
                fp.write(o)

    if len(nt):
        with open(out_path/'nt_sizes.yaml', 'w+') as fp:
            for o in set(nt):
                fp.write(o)

    if len(tn):
        with open(out_path/'tn_sizes.yaml', 'w+') as fp:
            for o in set(tn):
                fp.write(o)

    if len(tt):
        with open(out_path/'tt_sizes.yaml', 'w+') as fp:
            for o in set(tt):
                fp.write(o)

    print("[output]: {}".format(list(out_path.glob("*.yaml"))))

