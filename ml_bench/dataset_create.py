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


def add_feat(feat, solution):
    for k, v in solution.items():
        if k == 'ProblemType':
            for _k, _v in v.items():
                if isinstance(_v, list):
                    for i, o in enumerate(_v):
                        feat['PT_' + _k + f'_{i}'].append(o)
                else:
                    feat['PT_' + _k].append(strify(_v))
        elif isinstance(v, list):
            for i, o in enumerate(v):
                feat[k + f'_{i}'].append(o)
        else:
            feat[k].append(strify(v))


def dataset_create(basename:Path, test_pct=0.2, save_results=False, sampling_interval=1):
    print(f"{basename} ...")
    df = pd.read_csv(basename.with_suffix('.csv'))
    sol_start_idx = 10
    solution_names = df.columns[sol_start_idx:]
    problem_size_names = df.columns[1:sol_start_idx]
    gflops = df.iloc[:, sol_start_idx:].values
    rankings = np.argsort(-gflops) # reverse
    num_solutions = len(solution_names)

    train_features, test_features = defaultdict(lambda: []), defaultdict(lambda: [])
    _solutions = yaml.safe_load(basename.with_suffix('.yaml').open())
    problem_sizes, solutions = df[problem_size_names].values, _solutions[2:]
    assert len(solution_names) == len(solutions)
    total_col_set = set(get_parameter_names(solutions)) # total feature names

    train_idxs, test_idxs = split_idxs(len(problem_sizes), test_pct)
    assert(len(train_idxs) + len(test_idxs) == len(df))

	# raw
    for i, (problem_size, ranking) in enumerate(zip(problem_sizes, rankings)):
        features = train_features if i in train_idxs else test_features

        # train set can reduce sampling frequency
        if i in train_idxs:
            features = train_features
            n = list(range(0, num_solutions, sampling_interval))
        else:
            features = test_features
            n = list(range(num_solutions))

        for j in n:
            # problem sizes
            for k, v in zip(problem_size_names, problem_size):
                features[k.strip()].append(v)
            # solution features
            idx = ranking[j]
            solution, sname = solutions[idx], solution_names[idx]
            add_feat(features, solution)
            miss_cols = list(total_col_set - set(solution.keys()))
            for o in miss_cols: features[o].append(np.nan)
            features['SolutionName'].append(sname)
            features['GFlops'].append(gflops[i, idx])
            features['Ranking'].append(j / num_solutions)

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
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1])

    # get data
    src = []
    for f in list(path.glob("**/*.csv")):
        basename = f.parent/f.stem
        if os.path.exists(str(basename) + '.yaml'):
            src.append(basename)

    dfs, dfs2 = [], []
    # create train/test dataframe
#        with ThreadPoolExecutor(os.cpu_count()) as e:
#            df, df2 = e.map(dataset_create, src)
#            dfs.append(df)
#            dfs2.append(df2)
    for o in src:
        df, df2 = dataset_create(o, sampling_interval=2)
        dfs.append(df)
        dfs2.append(df2)

    train_df = pd.concat(dfs, ignore_index=True)
    train_df.to_csv(out/'train_raw_interval_2.csv', index=False)

    del dfs, train_df
    
    test_df = pd.concat(dfs2, ignore_index=True)
    test_df.to_csv(out/'test_raw_interval_2.csv', index=False)

    end = time.time()
    print("Prepare data done in {} seconds.".format(end - start))
