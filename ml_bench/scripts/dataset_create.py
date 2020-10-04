#!/usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import pdb
import yaml
import os
import math
import random
import pickle
import time
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def init_df(df):
    #df.fillna(0, inplace=True)
    df['PadA'] = df.apply(lambda x: x.LDA - x.SizeL if x.PT_TransposeA else x.LDA - x.SizeI, axis=1)
    df['PadB'] = df.apply(lambda x: x.LDB - x.SizeJ if x.PT_TransposeB else x.LDB - x.SizeL, axis=1)
    df['PadC'] = df['LDC'] - df['SizeI']
    df['PadD'] = df['LDD'] - df['SizeI']


def get_cols(df):
    cols = df.columns
    drop_cols = [o for o in cols if len(df[o].unique()) <= 1]
    return list(set(cols) - set(drop_cols))


def strify(o):
    if isinstance(o, list):
        return '_'.join([str(p) for p in o])
    elif isinstance(o, dict):
        return '_'.join(['{}:{}'.format(k, v) for k, v in o.items()])
    else:
        return o


def split_idxs(n, pct):
    n_test = max(int(n * pct), 1)
    n_train = n - n_test
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    return idxs[:n_train].copy(), idxs[n_train:].copy()


def df_create(features):
    df = pd.DataFrame()
    for k, v in features.items():
        df[k.strip()] = v
    return df


def get_parameter_names(solutions:dict):
    res = []
    for s in solutions:
        p = s.keys()
        if len(res) < len(p):
            res = p
    return list(res)


def dataset_create(basename:Path, test_pct=0.2, save_results=True):
    df = pd.read_csv(basename.with_suffix('.csv'))
    sol_start_idx = 10
    solution_names = df.columns[sol_start_idx:]
    problem_size_names = df.columns[1:sol_start_idx]
    gflops = df.iloc[:, sol_start_idx:].values
    ranking = gflops.argsort()

    train_features, test_features = defaultdict(lambda: []), defaultdict(lambda: [])
    problem_sizes = df[problem_size_names].values
    train_idxs, test_idxs = split_idxs(len(problem_sizes), test_pct)

    for row, ps in enumerate(problem_sizes):
        features = train_features if row in train_idxs else test_features
        for col, n in enumerate(solution_names):
            for k, v in zip(problem_size_names, ps):
                features[k.strip()].append(v)
            s = n.split('_')
            features['Layout'].append('_'.join(s[:3]))
            features['SolutionName'].append('_'.join(s[3:]))
            features['Ranking'].append(ranking[row, col])

    train_df = df_create(train_features)
    test_df = df_create(test_features)
    
    if save_results:
        train_df.to_csv(str(basename) + '_train_raw.csv', index=False)
        test_df.to_csv(str(basename) + '_test_raw.csv', index=False)

    return (train_df, test_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Create Dataset for ML_Bench of Tensile"
    parser.add_argument("--path", type=str, help='benchmark data dir')
    #parser.add_argument("--test", type=store_action, help='for test set')
    args = parser.parse_args()

    start = time.time()
    path = Path(args.path)

    # create train/test dataframe
    #with ThreadPoolExecutor(os.cpu_count()) as e:
    #    e.map(dataset_create, src)

    train_set, valid_set = [], []
    for o in path.glob("*.csv"):
        print("{} ...".format(o))
        tdf, vdf = dataset_create(o, 0.2, False)
        train_set.append(tdf)
        valid_set.append(vdf)
        print("done")

    print("train.csv ...")
    train_df = pd.concat(train_set, ignore_index=True)
    train_df.to_csv(path/'train.csv', index=False)
    print("done")
    
    print("valid.csv ...")
    valid_df = pd.concat(valid_set, ignore_index=True)
    valid_df.to_csv(path/'valid.csv', index=False)
    print("done")

    end = time.time()
    print("Prepare data done in {} seconds.".format(end - start))
