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


def df_merge(dfs):
    df = pd.concat(dfs, ignore_index=True)
    df['_UseSgprForGRO'] = df['_UseSgprForGRO'].replace('False', False).replace('1', True).replace('0', False).astype('bool')
    
    skip_cols = ['TotalFlops']
    for n, c in df.items():
        if is_integer_dtype(c) and n not in skip_cols:
            if c.max() < 128: df[n] = c.astype('int8')
            elif c.max() < 32768: df[n] = c.astype('int16')
            else: df[n] = c.astype('int32')

    for n, c in df.items():
        print(f"{n}: {c.dtype}: {c.unique() if not is_integer_dtype(c) else c.max()}")
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
    print(f"processing {basename}.csv ...")
    df = pd.read_csv(basename.with_suffix('.csv'))
    sol_start_idx = 10
    problem_size_names = df.columns[1:sol_start_idx]
    gflops = df.iloc[:, sol_start_idx:].values
    rankings = np.argsort(-gflops) # reverse

    train_features, test_features = defaultdict(lambda: []), defaultdict(lambda: [])
    _solutions = yaml.safe_load(basename.with_suffix('.yaml').open())
    problem_sizes, solutions = df[problem_size_names].values, _solutions[2:]

    solution_names = []
    with ThreadPoolExecutor(os.cpu_count()) as e:
        solution_names += e.map(Solution.getNameFull, solutions)
    #solution_names = [Solution.getNameFull(o) for o in solutions]
    num_solutions = len(solutions)
    print("num_solutions: {}\nsolution_names[0]: {}".format(num_solutions, solution_names[0]))

    total_col_set = set(get_parameter_names(solutions)) # total feature names

    train_idxs, test_idxs = split_idxs(len(problem_sizes), test_pct)
    assert(len(train_idxs) + len(test_idxs) == len(df))

    # raw
#    with ThreadPoolExecutor(os.cpu_count()) as e:
#        trn_feats, test_feats = parse_features()
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
            solution, (ptype, sname) = solutions[idx], solution_names[idx]
            add_feat(features, solution)
            miss_cols = list(total_col_set - set(solution.keys()))
            for o in miss_cols: features[o].append(np.nan)
            features['ProblemType'].append(ptype)
            features['SolutionName'].append(sname)
            features['GFlops'].append(gflops[i, idx])
            features['Ranking'].append(j / num_solutions)

    train_df = df_create(train_features)
    test_df = df_create(test_features)
    
    if save_results:
        train_df.to_csv(str(basename) + '_train_raw.csv', index=False)
        test_df.to_csv(str(basename) + '_test_raw.csv', index=False)
#        with open(str(basename) + '_solution_name.pkl', 'wb') as fp:
#            pickle.dump(solution_names, fp)

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
    sampling_interval = 1
#    # create train/test dataframe
#        with ThreadPoolExecutor(os.cpu_count()) as e:
#            df, df2 = e.map(dataset_create, src)
#            dfs.append(df)
#            dfs2.append(df2)
    for o in src:
        df, df2 = dataset_create(o, sampling_interval=sampling_interval)
        dfs.append(df)
        dfs2.append(df2)

    train_df = df_merge(dfs)
    test_df = df_merge(dfs2)

    # drop one-value columns
    to_keep = [n for n, c in train_df.items() if len(c.unique()) > 1]
    tail = 'raw_full' if sampling_interval == 1 else f'raw_sampling_interval_{sampling_interval}'
    train_df[to_keep].to_feather(out/f'train_{tail}.feat')
    print(f'{out}/train_{tail}.feat is generated.')

    test_df[to_keep].to_feather(out/f'test_{tail}.feat')
    print(f'{out}/test_{tail}.feat is generated.')

    end = time.time()
    print("Prepare data done in {} seconds.".format(end - start))
