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
#        return '_'.join([str(p) for p in o]) if o else '_'
        return '_'.join([str(p) for p in o])
    elif isinstance(o, dict):
        #return '_'.join(['{}:{}'.format(k, v) for k, v in o.items()]) if o else '_'
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

    train_features, test_features = defaultdict(lambda: []), defaultdict(lambda: [])
    _solutions = yaml.safe_load(basename.with_suffix('.yaml').open())
    problem_sizes, solutions = df[problem_size_names].values, _solutions[2:]
    assert len(solution_names) == len(solutions)
    col_set = set(get_parameter_names(solutions))

    train_idxs, test_idxs = split_idxs(len(problem_sizes), test_pct)

    for row, ps in enumerate(problem_sizes):
        features = train_features if row in train_idxs else test_features
        for col, (n, s) in enumerate(zip(solution_names, solutions)):
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
            #features['SolutionName'].append(n)
            features['GFlops'].append(max(df.iloc[row, col + sol_start_idx], 0))

    train_df = df_create(train_features)
    test_df = df_create(test_features)
    
    if save_results:
        train_df.to_csv(str(basename) + '_train_raw.csv', index=False)
        test_df.to_csv(str(basename) + '_test_raw.csv', index=False)
        with open(str(basename) + '_solution_name.pkl', 'wb') as fp:
            pickle.dump(solution_names, fp)

    return (train_df, test_df)


if __name__ == '__main__':
    start = time.time()
    path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data')

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
        df, df2 = dataset_create(o, 0.2, False)
        dfs.append(df)
        dfs2.append(df2)
        print("done")

    train_df = pd.concat(dfs, ignore_index=True)
    train_df.to_csv(out/'train_raw.csv', index=False)

    del dfs, train_df
    
    test_df = pd.concat(dfs2, ignore_index=True)
    test_df.to_csv(out/'test_raw.csv', index=False)

    end = time.time()
    print("Prepare data done in {} seconds.".format(end - start))
