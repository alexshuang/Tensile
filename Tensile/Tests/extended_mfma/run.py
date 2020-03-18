#!/usr/bin/env python3


import argparse
import json
import os
import pdb
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import shutil
import re


def save_file(data, out):
    yamls, gen_pass, run_pass, logs = data
    df = pd.DataFrame()
    #df['KernelName'] = np.array(kernel)
    df['Yaml'] = np.array(yamls)
    df['Pass'] = np.array(run_pass)
    df['GenPass'] = np.array(gen_pass)
    df['Error'] = np.array(logs)
    print("resoult save as " + out)
    df.to_csv(out, index=False)


def run(path, tensile_out, inst, threadtile, workgroup):
    gen_pass, run_pass, yamls, logs = [], [], [], []
    pat = re.compile(r'# 00_Final: End - ')

    tensile = "../../bin/Tensile"
    build_path = Path('build')
    build_path.mkdir(exist_ok=True)
    
    # copy yamls to build dir
    _path = build_path.joinpath(path.name)
    if not os.path.exists(_path):
        shutil.copytree(path, _path)
    path = _path

    # res dir
    res_path = path.joinpath('res')
    res_path.mkdir(exist_ok=True)

    m, n, k, b = inst.split('x')
    tt0, tt1 = threadtile.split('x')
    wg0, wg1, wg2 = workgroup.split('x')

    for f in path.glob("*.yaml"):
        yamls.append(f.name)
        res_f = res_path.joinpath(f.stem)

        # run tensile
        print("Running %s..." % f)
        ret = os.system("%s %s %s > %s" % (tensile, f, tensile_out, res_f))

        # resoult
        if ret:
            run_pass.append(False)
            res = res_f.open('r').readlines()
            tail_log = '\n'.join(res[-5:])
            if not len(pat.findall(tail_log)):
                gen_pass.append(False)
                logs.append(tail_log)
            else:
                gen_pass.append(True)
                logs.append('')
        else:
            run_pass.append(True)
            gen_pass.append(True)
            logs.append('')

    return yamls, gen_pass, run_pass, logs


if __name__ == "__main__":
    if not len(sys.argv) > 1:
        print("Usage: ./runner.py <path/to/test-case-dir> [Tensile output dir]")
        exit(1)

    inst = '32x32x1x2'
    threadtile = '1x32'
    workgroup = '16x16x1'
    path = Path(sys.argv[1])
    out_path = path.name + '.csv'
    tensile_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('output')

    res = run(path, tensile_out, inst, threadtile, workgroup)
    save_file(res, out_path)

