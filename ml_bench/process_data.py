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
from pandas.api.types import is_integer_dtype
from functools import partial
import argparse

pd.options.display.max_rows = None

datatype_properties = [
    {
        'char': 'S',
        'name': 'single',
        'nameAbbrev': 'f32',
        'enum': 'Float',
        'reg': 1,
        'ocl': 'float',
        'hip': 'float',
        'libType': 'float',
        'libEnum': 'tensileDataTypeFloat',
        'isIntegral': False,
        'isComplex': False,
        'packing': 1,
        'miInput' : 1,
    },
    {
        'char': 'D',
        'name': 'double',
        'nameAbbrev': 'f64',
        'enum': 'Double',
        'reg': 2,
        'ocl': 'double',
        'hip': 'double',
        'libType': 'double',
        'libEnum': 'tensileDataTypeDouble',
        'isIntegral': False,
        'isComplex': False,
        'packing': 1,
        'miInput' : 1,
    },
    {
        'char': 'C',
        'name': 'complexSingle',
        'nameAbbrev': 'f32c',
        'enum': 'ComplexFloat',
        'reg': 2,
        'ocl': 'float2',
        'hip': 'TensileComplexFloat',
        'libType': 'TensileComplexFloat',
        'libEnum': 'tensileDataTypeComplexFloat',
        'isIntegral': False,
        'isComplex': True,
        'packing': 1,
        'miInput' : 1,
    },
    {
        'char': 'Z',
        'name': 'complexDouble',
        'nameAbbrev': 'f64c',
        'enum': 'ComplexDouble',
        'reg': 4,
        'ocl': 'double2',
        'hip': 'TensileComplexDouble',
        'libType': 'TensileComplexDouble',
        'libEnum': 'tensileDataTypeComplexDouble',
        'isIntegral': False,
        'isComplex': True,
        'packing': 1,
        'miInput' : 1,
    },
    {
        'char': 'H',
        'name': 'half',
        'nameAbbrev': 'f16',
        'enum': 'Half',
        'reg': 0.5,
        'ocl': 'ERROR',
        'hip': 'tensile_half',
        'libType': 'TensileHalf',
        'libEnum': 'tensileDataTypeHalf',
        'isIntegral': False,
        'isComplex': False,
        'packing': 1,
        'miInput' : 4,
    },
    {
        'char': '4xi8',
        'name': 'int8x4',
        'nameAbbrev': 'i8',
        'enum': 'Int8x4',
        'reg': 1,
        'ocl': 'ERROR',
        'hip': 'uint32_t',
        'libType': 'TensileInt8x4',
        'libEnum': 'tensileDataTypeInt8x4',
        'isIntegral': True,
        'isComplex': False,
        'packing': 4,
        'miInput' : 4,
    },
    {
        'char': 'I',
        'name': 'int32',
        'nameAbbrev': 'i32',
        'enum': 'Int32',
        'reg': 1,
        'ocl': 'ERROR',
        'hip': 'int32_t',
        'libType': 'TensileInt32',
        'libEnum': 'tensileDataTypeInt32',
        'isIntegral': True,
        'isComplex': False,
        'packing': 1,
        'miInput' : 1,
    },
    {
        'char': 'B',
        'name': 'bfloat16',
        'nameAbbrev': 'bf16',
        'enum': 'BFloat16',
        'reg': 0.5,
        'ocl': 'ERROR',
        'hip': 'tensile_bfloat16',
        'libType': 'tensile_bfloat16',
        'libEnum': 'tensileDataTypeBFloat16',
        'isIntegral': False,
        'isComplex': False,
        'packing': 1,
        'miInput' : 2,
    },
]


class ProblemType():
  @ staticmethod
  def getName(state):
    indexChars = "IJKLMNOPQRSTUVWXYZ"  # which characters to use for C[ij]=Sum[k] A[ik]*B[jk]
    # C dimensions
    name = "C"
    for i in range(0, state["NumIndicesC"]):
      name += indexChars[i].lower()
    # A dimensions
    name += "_A"
    for i in state["IndexAssignmentsA"]:
      name += indexChars[i].lower()
    if state["ComplexConjugateA"]:
      name += "C"
    # B dimensions
    name += "_B"
    for i in state["IndexAssignmentsB"]:
      name += indexChars[i].lower()
    if state["ComplexConjugateB"]:
      name += "C"

    # precision and other
    name += "_"
    name += datatype_properties[state["DataType"]]['char']
    if state["UseBeta"]: name += "B"
    if state["HighPrecisionAccumulate"] and not state["SilentHighPrecisionAccumulate"]: name += "H"
    if state["UseInitialStridesAB"]: name += "I"
    if state["UseInitialStridesCD"]: name += "Ic"
    return name


class Solution():
  @ staticmethod
  def getNameFull(state):
    return (ProblemType.getName(state['ProblemType']), Solution.getNameMin(state))

  @ staticmethod
  def getParameterNameAbbreviation( name ):
    return ''.join([c for c in name if not c.islower()])

  @ staticmethod
  def getParameterValueAbbreviation( key, value ):
    if isinstance(value, str):
      return ''.join([c for c in value if c.isupper()])
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      if value >= 0:
        return "%u" % value
      else: # -1 -> n1
        return "n%01u" % abs(value)
    elif isinstance(value, ProblemType):
      return str(value)
    elif isinstance(value, tuple) or key == 'ISA':
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += str(value[i])
      return abbrev
    elif isinstance(value, list):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += Solution.getParameterValueAbbreviation(key, value[i])
        if i < len(value)-1:
          abbrev += "_"
      return abbrev
    elif isinstance(value, dict):
      s =  "_".join(["%d%d"%(pos,k) for pos,k in value.items()])
      return s
    else:
      printExit("Parameter \"%s\" is new object type" % str(value) )
      return str(value)

  @ staticmethod
  def getNameMin(state):
    name = ""
    first = True
    # put problem first
#    if "ProblemType" in state:
#      name += str(state["ProblemType"]) + "_"
    if "MacroTile0" in state \
        and "MacroTile1" in state \
        and "DepthU" in state:
      name += "%s%ux%ux%u_" \
          % ( Solution.getParameterNameAbbreviation("MacroTile"), \
          state["MacroTile0"], state["MacroTile1"], state["DepthU"] )
    if "MatrixInstM" in state:
      name += "%s%ux%ux%ux%u_" \
          % ( Solution.getParameterNameAbbreviation("MatrixInstruction"), \
          state["MatrixInstM"], state["MatrixInstN"], state["MatrixInstK"], state["MatrixInstB"])
    if "LdcEqualsLdd" in state:
      if state["LdcEqualsLdd"]:
        name += "SE_"
      else:
        name += "SN_"
    for key in sorted(state.keys()):
      if key != 'ProblemType' and key[0] != '_':
        if not first:
          name += "_"
        else:
          first = False
        name += "%s%s" % ( Solution.getParameterNameAbbreviation(key), \
                Solution.getParameterValueAbbreviation(key, state[key]) )
    return name


def strify(o):
    if isinstance(o, list):
        return '_'.join([str(p) for p in o])
    elif isinstance(o, dict):
        return '_'.join(['{}:{}'.format(k, v) for k, v in o.items()])
    else:
        return o


def split_idxs(n, pct):
    n_valid = max(int(n * pct), 1)
    n_train = n - n_valid
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    return idxs[:n_train].copy(), idxs[n_train:].copy()


def df_create(features):
    df = pd.DataFrame()
    for k, v in features.items():
        df[k.strip()] = v
    return df


def df_compress(df):
    df['_UseSgprForGRO'] = df['_UseSgprForGRO'].replace('False', False).replace('1', True).replace('0', False).astype('bool')
    skip_cols = ['TotalFlops']
    for n, c in df.items():
        if is_integer_dtype(c) and n not in skip_cols:
            if c.max() < 128: df[n] = c.astype('int8')
            elif c.max() < 32768: df[n] = c.astype('int16')
            else: df[n] = c.astype('int32')


def df_merge(dfs):
    df = pd.concat(dfs, ignore_index=True)
    df_compress(df)
    
    #for n, c in df.items():
    #    print(f"{n}: {c.dtype}: {c.unique() if not is_integer_dtype(c) else c.max()}")
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


def feature_parse(idx, problem_size_names, solution_names, problem_sizes,
        solutions, rankings, train_idxs, gflops, sampling_interval=1):
    problem_size, ranking = problem_sizes[idx], rankings[idx]
    ds_type = 'train' if idx in train_idxs else 'valid'
    features = defaultdict(lambda: [])
    num_solutions = len(solution_names)
    total_col_set = set(get_parameter_names(solutions)) # total feature names

    # train set can reduce sampling frequency
    if ds_type == 'train':
        n = range(0, num_solutions, sampling_interval)
    else:
        n = range(num_solutions)

    for j in n:
        # problem sizes
        for k, v in zip(problem_size_names, problem_size):
            features[k.strip()].append(v)
        # solution features
        r = ranking[j]
        #solution, (ptype, sname) = solutions[r], solution_names[r]
        solution, sname = solutions[r], solution_names[r]
        add_feat(features, solution)
        miss_cols = list(total_col_set - set(solution.keys()))
        for o in miss_cols: features[o].append(np.nan)
        #features['ProblemType'].append(ptype)
        features['SolutionName'].append(sname)
        features['GFlops'].append(gflops[idx, r])
        features['Ranking'].append(j / num_solutions)
    return (idx if ds_type == 'valid' else None, ds_type, features)


def dataset_create(basename:Path, valid_pct=0.2, sampling_interval=1, n_jobs=-1, is_test=False):
    print(f"processing {basename} ...")
    df = pd.read_csv(basename.with_suffix('.csv'))
    sol_start_idx = 10
    problem_size_names = df.columns[1:sol_start_idx]
    gflops = df.iloc[:, sol_start_idx:].values
    rankings = np.argsort(-gflops) # reverse
    if n_jobs == -1: n_jobs = os.cpu_count()
    workdir = basename.parent

    features = { 'train': defaultdict(lambda: []),
                 'valid': defaultdict(lambda: []) }
    _solutions = yaml.safe_load(basename.with_suffix('.yaml').open())
    problem_sizes, solutions = df[problem_size_names].values, _solutions[2:]

    num_solutions = len(solutions)
    solution_names = df.columns[sol_start_idx:].values # min name
    # get full name 
    #solution_names = []
    #with ThreadPoolExecutor(n_jobs) as e:
    #    solution_names += e.map(Solution.getNameFull, solutions)
    #print("num_solutions: {}\nsolution_names[0]: {}".format(num_solutions, solution_names[0]))

    train_idxs, valid_idxs = split_idxs(len(problem_sizes), valid_pct)
    assert(len(train_idxs) + len(valid_idxs) == len(df))

    # parse features
    feats, valid_config_indices = [], []
    with ThreadPoolExecutor(n_jobs) as e:
        feats += e.map(partial(feature_parse,
                            problem_size_names=problem_size_names,
                            solution_names=solution_names,
                            problem_sizes=problem_sizes,
                            solutions=solutions,
                            rankings=rankings,
                            train_idxs=train_idxs,
                            gflops=gflops,
                            sampling_interval=sampling_interval
                            ),
                            np.arange(len(problem_sizes)))

    for idx, n, feat in feats:
        for k, v in feat.items():
            features[n][k].extend(v)
        if idx is not None:
            valid_config_indices.append(idx)

    train_df = df_create(features['train'])
    valid_df = df_create(features['valid'])
    
    if not is_test:
        configs = (workdir/'problem_sizes.yaml').open().readlines()
        valid_df.to_csv(workdir/f'valid_N{num_solutions}.csv', index=False)
        with (workdir/'valid_problem_sizes.yaml').open('w') as fp:
            for i in valid_config_indices: fp.write(configs[i])
    else:
        df = pd.concat([train_df, valid_df], ignore_index=True)
        df_compress(df)
        df.to_feater(workdir/f'test_N{num_solutions}.feat', index=False)

    return (train_df, valid_df)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.description = "process tensile benchmark data"
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--is_test", action="store_true", default=None)
    parser.add_argument("--sampling_interval", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    path = Path(args.data_dir)
    out = Path(args.output_dir) if args.output_dir else path
    out.mkdir(exist_ok=True)

    # get data
    src = []
    for f in list(path.glob("**/*.csv")):
        basename = f.parent/f.stem
        if os.path.exists(str(basename) + '.yaml'):
            src.append(basename)

    dfs, dfs2 = [], []
    for o in src:
        df, df2 = dataset_create(o, sampling_interval=args.sampling_interval, n_jobs=args.n_jobs, is_test=args.is_test)
        dfs.append(df)
        dfs2.append(df2)

    if not args.is_test:
        train_df = df_merge(dfs)
        valid_df = df_merge(dfs2)

        tail = '' if args.sampling_interval == 1 else f'_sampling_interval_{args.sampling_interval}'
        # drop one-value columns
        to_keep = [n for n, c in train_df.items() if len(c.unique()) > 1]
        train_df[to_keep].to_feather(out/f'train{tail}.feat')
        print(f'{out}/train{tail}.feat is generated.')
        valid_df[to_keep].to_feather(out/f'valid{tail}.feat')
        print(f'{out}/valid{tail}.feat is generated.')
        #df = pd.concat([train_df[to_keep], valid_df[to_keep]], ignore_index=True)
        #df.to_feather(out/f'train_and_valid{tail}.feat')
        #print(f'{out}/train_and_valid{tail}.feat is generated.')

    end = time.time()
    print("Prepare data done in {} seconds.".format(end - start))

