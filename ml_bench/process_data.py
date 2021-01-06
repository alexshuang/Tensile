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
import re

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


def df_create(problem_features, kernel_features, bench_features, num_problems):
    df = pd.DataFrame()
    for k, v in problem_features.items():
        df[k.strip()] = v
    for k, v in kernel_features.items():
        df[k.strip()] = v * num_problems
    for k, v in bench_features.items():
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
    return df


def parse_kernel_feature(kernels):
    feat = defaultdict(lambda: [])
    for kernel in kernels:
        for k, v in kernel.items():
            if k == 'ProblemType':
                for _k, _v in v.items():
                    if isinstance(_v, list):
                        for i, o in enumerate(_v):
                            feat['PT_' + _k.strip() + f'_{i}'].append(o)
                    else:
                        feat['PT_' + _k.strip()].append(strify(_v))
            elif isinstance(v, list):
                for i, o in enumerate(v):
                    feat[k.strip() + f'_{i}'].append(o)
            else:
                feat[k.strip()].append(strify(v))
    return feat


def get_rankings(gflops, num_kernels):
    itor = {r:i for i, r in enumerate(np.argsort(-gflops))} # reverse
    rankings = [itor[j] / num_kernels for j in range(num_kernels)]
    return rankings


def parse_config(conf):
    global_param_start_offset = -1
    problem_size_start_offset = -1
    text, problem_size = [], []
    gp_start_pat = r'^GlobalParameters\s*:'
    ps_start_pat = r'\s*-\s*ProblemSizes\s*'
    ps_pat = r'\s*-\s*Exact\s*:\s*[\[\{]'
    for i, o in enumerate(conf.open().readlines()):
        if re.match(ps_pat, o):
            problem_size.append(o)
        else:
            if re.match(gp_start_pat, o): global_param_start_offset = i + 1
            elif re.match(ps_start_pat, o): problem_size_start_offset = i + 1
            text.append(o)
    return global_param_start_offset, problem_size_start_offset, text, problem_size


def dataset_create(basename, valid_pct=0.2, test=False):
    print(f"processing {basename} ...")
    df = pd.read_csv(basename.with_suffix('.csv'))
    sol_start_idx = 10
    problem_size_names = df.columns[1:sol_start_idx]
    gflops = df.iloc[:, sol_start_idx:].values
    rankings = np.argsort(-gflops) # reverse
    workdir = basename.parent

    _kernels = yaml.safe_load(basename.with_suffix('.yaml').open())
    problem_sizes, kernels = df[problem_size_names].values, _kernels[2:]
    num_problems, num_kernels = len(problem_sizes), len(kernels)
    kernel_names = df.columns[sol_start_idx:].values # min name
    
    # get full name 
    #solution_names = []
    #if n_jobs == -1: n_jobs = os.cpu_count()
    #with ThreadPoolExecutor(n_jobs) as e:
    #    solution_names += e.map(Solution.getNameFull, solutions)
    #print("num_solutions: {}\nsolution_names[0]: {}".format(num_solutions, solution_names[0]))

    kernel_features = parse_kernel_feature(kernels)
    kernel_features['KernelName'].extend(kernel_names)

    if not test:
        train_idxs, valid_idxs = split_idxs(num_problems, valid_pct)
        assert(len(train_idxs) + len(valid_idxs) == len(df))
        train_problems, train_gflops, train_rankings = [], [], []
        valid_problems, valid_gflops, valid_rankings = [], [], []
        for i in train_idxs:
            train_problems.append(problem_sizes[i])
            train_gflops.append(gflops[i])
            train_rankings.append(get_rankings(gflops[i], num_kernels))
        for i in valid_idxs:
            valid_problems.append(problem_sizes[i])
            valid_gflops.append(gflops[i])
            valid_rankings.append(get_rankings(gflops[i], num_kernels))

        train_problem_features = defaultdict(lambda: [])
        for n, v in zip(problem_size_names, np.transpose(train_problems)):
            train_problem_features[n].extend(np.repeat(v, num_kernels))
        valid_problem_features = defaultdict(lambda: [])
        for n, v in zip(problem_size_names, np.transpose(valid_problems)):
            valid_problem_features[n].extend(np.repeat(v, num_kernels))

        train_bench_features = {
            'GFlops': np.concatenate(train_gflops),
            'Ranking': np.concatenate(train_rankings),
        }
        valid_bench_features = {
            'GFlops': np.concatenate(valid_gflops),
            'Ranking': np.concatenate(valid_rankings),
        }

        train_df = df_create(train_problem_features, kernel_features, train_bench_features, len(train_problems))
        valid_df = df_create(valid_problem_features, kernel_features, valid_bench_features, len(valid_problems))
            
        # validate
        _valid_df = valid_df.copy()
        df_compress(_valid_df)
        _valid_df.to_feather(workdir/f'valid_N{num_kernels}.feat')
        del _valid_df

        # config
        config = list(workdir.glob('rocblas_*.yaml'))[0]
        valid_output = workdir/f'valid_{config.stem}.yaml'
        fast_bench_output = workdir/f'fast_bench_{config.stem}.yaml'
        global_param_start_offset, problem_size_start_offset, text, problem_size = parse_config(config)
        valid_problem_size = [problem_size[i] for i in valid_idxs]
        for o in reversed(valid_problem_size): text.insert(problem_size_start_offset, o)
        with (valid_output).open('w') as fp:
            fp.write(''.join(text))
        print(f"{valid_output} is generated.")
        text.insert(global_param_start_offset, '  FastSolutionKeep: 0.25\n')
        text.insert(global_param_start_offset, '  FastBenchmark: True\n')
        with (fast_bench_output).open('w') as fp:
            fp.write(''.join(text))
        print(f"{fast_bench_output} is generated.")
    else:
        problem_features = defaultdict(lambda: [])
        for n, v in zip(problem_size_names, np.transpose(problem_sizes)):
            problem_features[n].extend(np.repeat(v, num_kernels))
        rankings = [get_rankings(gflops[i], num_kernels) for i in range(num_problems)]
        bench_features = {
            'GFlops': np.concatenate(gflops),
            'Ranking': np.concatenate(rankings),
        }
        df = df_create(problem_features, kernel_features, bench_features, num_problems)
        df_compress(df)
        df.to_feather(workdir/f'test_N{num_kernels}.feat')
        # config
        config = list(workdir.glob('rocblas_*.yaml'))[0]
        fast_bench_output = workdir/f'fast_bench_{config.stem}.yaml'
        with (fast_bench_output).open('w') as fp:
            gp_start_pat = r'^GlobalParameters\s*:'
            for o in config.open().readlines():
                fp.write(o)
                if re.match(gp_start_pat, o):
                    fp.write('  FastBenchDebug: True\n')
                    fp.write('  FastSolutionPct: 0.25\n')
                    fp.write('  FastBenchmark: True\n')
        print(f"{fast_bench_output} is generated.")
        return (None, None)

    return (train_df, valid_df)


def _process_data(args):
    path = Path(args.path)
    out = path
    out.mkdir(exist_ok=True)

    start = time.time()

    # get data
    src = []
    for f in list(path.glob("**/*.csv")):
        basename = f.parent/f.stem
        if os.path.exists(str(basename) + '.yaml'):
            src.append(basename)

    train_dfs, valid_dfs = [], []
    for o in src:
        df, df2 = dataset_create(o, valid_pct=args.valid_pct, test=args.test)
        train_dfs.append(df)
        valid_dfs.append(df2)

    if not args.test:
        train_df = df_merge(train_dfs)
        valid_df = df_merge(valid_dfs)

        # drop one-value columns
        to_keep = [n for n, c in train_df.items() if len(c.unique()) > 1]
        train_df, valid_df = train_df[to_keep], valid_df[to_keep]
        train_df.to_feather(out/f'train.feat')
        print(f'{out}/train.feat is generated.')
        valid_df.to_feather(out/f'valid.feat')
        print(f'{out}/valid.feat is generated.')
        if args.train_and_valid:
            df = pd.concat([train_df, valid_df], ignore_index=True)
            df.to_feather(out/f'train_and_valid.feat')
            print(f'{out}/train_and_valid.feat is generated.')

    end = time.time()
    print("Prepare data done in {} seconds.".format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Convert benchmark data into what the model needs."
    parser.add_argument("--path", type=str, help="Path of workspace.")
    parser.add_argument("--valid_pct", type=float, default=0.2, help="The proportion of validation sets in the entire data set.")
    parser.add_argument("--train_and_valid", action="store_true", default=False, help="Merge train.feat and valid.feat to train_and_valid.feat")
    parser.add_argument("--test", action="store_true", default=False, help="Process for test set. Compared with train set, test set has no label and does not need to split problem size for validation set.")
    args = parser.parse_args()

    _process_data(args)
