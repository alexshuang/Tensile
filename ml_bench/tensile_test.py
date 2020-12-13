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
parser.description = "tensile testing"
parser.add_argument('-p', "--prediction", type=str)
parser.add_argument('-t', "--target", type=str)
parser.add_argument('-m', "--model_dir", type=str)
parser.add_argument('-o', "--output_dir", type=str, default=None)
args = parser.parse_args()

out_path = Path(args.output_dir)
img_path = out_path/'imgs'
model_path = Path(args.model_dir)
out_path.mkdir(exist_ok=True)
img_path.mkdir(exist_ok=True)

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

def testing(pred_path, target_path, topN=5, n_pct=0.15, log=False):
    pred_df = pd.read_csv(pred_path, low_memory=False)
    target_df = pd.read_csv(target_path, low_memory=False)

    cols = pred_df.columns

    pred_df.replace(" " , '0', inplace=True)
    for n, c in pred_df[cols[10:]].items():
        if is_object_dtype(c):
            pred_df[n] = pd.to_numeric(c).astype('float')

    gflops_preds = pred_df[cols[10:]].values
    n, m = gflops_preds.shape
    num_preds = int(m * n_pct)
    topN_preds = np.argsort(-gflops_preds)[:, :num_preds]

    gflops = target_df[cols[10:]].values
    topN_target = np.argsort(-gflops)[:, :topN]

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

    print(f"{pred_path}: {n} problems, {n_pct*100}%/{num_preds} solutions, top1 accuracy: {np.sum(top1_acc)/n*100:.2f}%, top{topN} accuracy: {np.sum(acc)/n*100:.2f}%")
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
    plt.gcf().savefig(img_path/f'{Path(pred_path).stem}_problem{n}_pct{n_pct}_top{topN}.png', dpi=600, bbox_inches='tight')


print("Loading model ...")
start = time.time()
model = pickle.load((model_path/'final_rf_model.pkl').open('rb'))
final_cols = pickle.load((model_path/'final_columns.pkl').open('rb'))
end = time.time()
print("done in {:.2f} seconds.".format(end - start))

print("Testing model ...")
testing(args.prediction, args.target, n_pct=0.15, log=True)

print("Done")
