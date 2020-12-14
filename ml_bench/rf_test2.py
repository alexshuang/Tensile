#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_bool_dtype, is_numeric_dtype, is_object_dtype, is_categorical_dtype, is_integer_dtype, is_float_dtype
# from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
#from IPython.display import Image, display_svg, SVG
# from dtreeviz.trees import *
from sklearn.tree import export_graphviz
import scipy
from scipy.cluster import hierarchy as hc
from sklearn.inspection import plot_partial_dependence
import yaml
import os
import re
import math
import random
import pickle
import time
from tqdm import tqdm
from collections import defaultdict
from sklearn import base
from sklearn.model_selection import KFold
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.description = "model testing"
parser.add_argument('-d', "--data_dir", type=str)
parser.add_argument('-m', "--model_dir", type=str)
parser.add_argument('-o', "--output_dir", type=str, default=None)
parser.add_argument('-n', "--n_pct", type=float, default=0.15)
args = parser.parse_args()

path = Path(args.data_dir)
img_path = path/'imgs'
model_path = Path(args.model_dir)
img_path.mkdir(exist_ok=True)
model_path.mkdir(exist_ok=True)
out_path = path if args.output_dir is None else Path(args.output_dir)

def rmse(pred,y):
    return round(math.sqrt(((pred-y)**2).mean()), 4)

def m_rmse(m, xs, y):
    return rmse(m.predict(xs), y)

def eval_model(m, trn_xs, trn_y, val_xs, val_y):
    return m_rmse(m, trn_xs, trn_y), m_rmse(m, val_xs, val_y)

def mae(p, t): return np.mean(abs(p - t))

def train_cats(df):
    for n, c in df.items():
        if is_object_dtype(c):
            df[n] = c.astype('category').cat.as_ordered()

def categorify(df):
    for n, c in df.items():
        if is_bool_dtype(c):
            df[n] = c.astype('int8')
        elif not is_numeric_dtype(c):
            df[n] = pd.Categorical(c).codes + 1

def preproc_df(df):
#     nn_idxs = np.where(df['ProblemType'].apply(lambda x: 'Ailk_Bljk' in x))
#     tn_idxs = np.where(df['ProblemType'].apply(lambda x: 'Alik_Bljk' in x))
#     nt_idxs = np.where(df['ProblemType'].apply(lambda x: 'Ailk_Bjlk' in x))
#     df.loc[nn_idxs]['PadA'] = df['LDA'] - df['SizeI']
#     df.loc[nn_idxs]['PadB'] = df['LDB'] - df['SizeL']
    df['PadC'] = df['LDC'] - df['SizeI']
#     df.loc[tn_idxs]['PadA'] = df['LDA'] - df['SizeL']
#     df.loc[nt_idxs]['PadB'] = df['LDB'] - df['SizeJ']
    df['AspectRatioA'] = (df['SizeL'] / df['SizeI']).astype('float32')
    df['AspectRatioB'] = (df['SizeJ'] / df['SizeL']).astype('float32')
    df['AspectRatioC'] = (df['SizeJ'] / df['SizeI']).astype('float32')
    df['AreaA'] = (df['SizeI'] * df['SizeL']).astype('int64')
    df['AreaB'] = (df['SizeJ'] * df['SizeL']).astype('int64')
    df['AreaC'] = (df['SizeI'] * df['SizeJ']).astype('int64')
    df['AoverB'] = (df['AreaA'] / df['AreaB']).astype('float32')
#     df['TotalFlops'] = df['TotalFlops'] / 1e9
    dup_cols = ['LDD', 'MacroTileA', 'MacroTileB','SubGroupA', 'SubGroupB',
                'ThreadTileA', 'ThreadTileB', 'MatrixInstBM', 'MatrixInstN']
    df.drop(dup_cols, axis=1, inplace=True)

def testing(test_csv, ns_pat, n_pct=0.1, topN=5, log=False):
    for f in test_csv:
        num_solution = re.findall(ns_pat, f.stem)[0]
        assert num_solution.isdecimal()
        num_solution = eval(num_solution)

        df = pd.read_feather(f)
        gflops = df['GFlops'].values
        gflops = gflops.reshape(-1, num_solution)
        n = gflops.shape[0]
        topN_target = np.argsort(-gflops)[:, :topN]

        preproc_df(df)
        train_cats(df)
        categorify(df)
        test_xs = df[final_cols]
        print("dump test_xs ...")
        test_xs.to_feather(out_path/"test_xs.feat")
        if log: start = time.time()
        preds = model.predict(test_xs)
        if log: print("model inference done in {:.2f} seconds.".format(time.time() - start))
        preds = np.expm1(preds)
        preds = preds.reshape(gflops.shape)
        num_preds = int(preds.shape[1] * n_pct)
        topN_preds = np.argsort(preds)[:, :num_preds]

        if log:
            print("Dumping fast_solution_indices.csv ...")
            with (out_path/f'fast_solution_indices.csv').open('w') as fp:
                for p in topN_preds:
                    fp.write(f"{','.join([str(o) for o in np.sort(p)])}\n")

        top1_acc, acc = [], []
        for p, t in zip(topN_preds, topN_target):
            if t[0] in p:
                top1_acc.append(True)
                acc.append(True)
            else:
                for o in t[1:]:
                    if o in p:
                        acc.append(True)
                        break

        gflops_preds, gflops_target = [], []
        for i, (p, t) in enumerate(zip(topN_preds, topN_target[:, 0].reshape(-1))):
            max_gflops = 0
            for j in p:
                if gflops[i, j] > max_gflops:
                    max_gflops = gflops[i, j]
            gflops_preds.append(max_gflops)
            gflops_target.append(gflops[i, t])
        gflops_preds, gflops_target = np.array(gflops_preds), np.array(gflops_target)

        print(f"{f.parent.stem}: {n} problems, {n_pct*100}%/{num_preds} solutions, top1 accuracy: {np.sum(top1_acc)/n*100:.2f}%, top{topN} accuracy: {np.sum(acc)/n*100:.2f}%")
        print(f"\t\ttotal errors: {abs(gflops_preds - gflops_target).sum():.2f} GFlops, mean errors: {mae(gflops_preds, gflops_target):.2f} GFlops")
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        x_axis = np.arange(n)
        j = n // 9
        for i, ax in enumerate(axes.flatten()):
            ax.plot(x_axis[i*j:i*j+j], gflops_preds[i*j:i*j+j], label='preds')
            ax.plot(x_axis[i*j:i*j+j], gflops_target[i*j:i*j+j], label='target')
        plt.subplots_adjust(right=1.5)
        plt.legend()
        plt.show()
        plt.gcf().savefig(img_path/f'{f.parent.stem}_problem{n}_pct{n_pct}_top{topN}.png', dpi=600, bbox_inches='tight')


print("Loading model ...")
start = time.time()
model = pickle.load((model_path/'final_rf_model.pkl').open('rb'))
final_cols = pickle.load((model_path/'final_columns.pkl').open('rb'))
end = time.time()
print("done in {:.2f} seconds.".format(end - start))

test_csv = list(path.glob('**/test_N*.feat'))
ns_pat = re.compile(r'_N(.*)')

print("Testing model ...")
testing(test_csv, ns_pat, n_pct=args.n_pct, log=True)

print("Done")
