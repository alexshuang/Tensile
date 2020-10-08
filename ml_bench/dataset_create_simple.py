import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
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
    num_solutions = len(solution_names)

    train_features, test_features = defaultdict(lambda: []), defaultdict(lambda: [])
    _solutions = yaml.safe_load(basename.with_suffix('.yaml').open())
    problem_sizes, solutions = df[problem_size_names].values, np.array(_solutions[2:])
    assert len(solution_names) == len(solutions)
    col_set = set(get_parameter_names(solutions))
    gflops = df.iloc[:, sol_start_idx:].values
    rankings = gflops.argsort()

    train_idxs, test_idxs = split_idxs(len(problem_sizes), test_pct)

    for row, (rnk, ps) in enumerate(zip(rankings, problem_sizes)):
        sol_sorted = solutions[rnk]
        if row in train_idxs:
            features = train_features
            is_train = True
        else:
            features = test_features
            is_train = False

        if is_train:
            n = list(range(0, num_solutions, 5)) + [num_solutions - 2, num_solutions - 1]
        else:
            n = list(range(num_solutions))
            
        for col in n:
            s = sol_sorted[col]
            for k, v in zip(problem_size_names, ps):
                features[k.strip()].append(v)
            for k, v in s.items():
                if k == 'ProblemType':
                    for _k, _v in v.items():
                        features['PT_' + _k].append(strify(_v))
                else:
                    features[k].append(strify(v))
            miss_cols = list(col_set - set(s.keys()))
            for o in miss_cols:
                features[o].append(np.nan)
            features['GFlops'].append(gflops[row, col])

    train_df = df_create(train_features)
    test_df = df_create(test_features)
    
    train_df.drop_duplicates(inplace=True)
    train_df = train_df[~(train_df.GFlops < 0)].reset_index(drop=True)
    test_df.drop_duplicates(inplace=True)
    test_df = test_df[~(test_df.GFlops < 0)].reset_index(drop=True)

    if save_results:
        train_df.to_csv(str(basename) + '_train_raw_simple.csv', index=False)
        test_df.to_csv(str(basename) + '_test_raw_simple.csv', index=False)

    return (train_df, test_df)


if __name__ == '__main__':
    start = time.time()
    path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1])

    # get data
    src = []
    for f in list(path.glob("**/*.csv")):
        basename = f.parent/f.stem
        if os.path.exists(str(basename) + '.yaml'):
            src.append(basename)

    # create train/test dataframe
    #with ThreadPoolExecutor(os.cpu_count()) as e:
    #    e.map(dataset_create, src)
    dfs, dfs2 = [], []
    for o in src:
        print("{} ...".format(o))
        train_df, valid_df = dataset_create(o, 0.2, False)
        dfs.append(train_df)
        dfs2.append(valid_df)
        print("done")

    train_df = pd.concat(dfs, ignore_index=True)
    train_df.to_csv(out/'train_raw_simple.csv', index=False)

    #del dfs, train_df
    
    test_df = pd.concat(dfs2, ignore_index=True)
    test_df.to_csv(out/'test_raw_simple.csv', index=False)

    end = time.time()
    print("Prepare data done in {} seconds.".format(end - start))
