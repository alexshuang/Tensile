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


# ## Look at Data

# In[2]:


path = Path('data/inc1/bench')
nn_path = path/'nn'
nt_path = path/'nt'
tn_path = path/'tn'
img_path = path/'imgs'
model_path = Path('data/inc1/models')
img_path.mkdir(exist_ok=True)
model_path.mkdir(exist_ok=True)


# In[6]:


def classify_df(df):
    for n, c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
            
def apply_cats(df, src_df):
    for n, c in df.items():
        if n in src_df.columns and src_df[n].dtype.name == 'category':
            df[n] = df[n].astype('category').cat.as_ordered()
            df[n].cat.set_categories(src_df[n].cat.categories, ordered=True, inplace=True)
    
def tolist(x):
    if isinstance(x, str): return [x]
    elif isinstance(x, tuple): return list(x)
    return x

def tostr(x):
    if isinstance(x, [list, tuple]): return '_'.join(x)
    return x

def kfold_target_encoding(train_df, recipies, n_fold=5):
    tme_cols = []
    train_new = train_df.copy()
    for i, (gby_col, target_col, op) in enumerate(recipies):
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=21)
        for tr_ind, val_ind in kf.split(train_df):
            trn_df, val_df = train_df.iloc[tr_ind], train_df.iloc[val_ind]
            agg = trn_df.groupby(gby_col)[target_col].agg(op)
            col_names = ['_'.join([gby_col, c, 'target']) for c in agg.columns]
            agg.columns = col_names
            for c in col_names:
                train_new.loc[val_ind, c] = val_df[gby_col].map(agg[c])
            tme_cols.extend(col_names)
        train_new.fillna(train_df[target_col].median(), inplace=True)
    return train_new, np.unique(tme_cols)

def gen_tme_feat(trn_df):
    agg_op = ['mean', 'median']
    dep_var = 'GFlops'
    recipies = [
        ('SolutionName', dep_var, agg_op),
    ]
    return kfold_target_encoding(trn_df, recipies)

# apply target mean encoding by train dataframe
def apply_tme_feat(df, train_df, tme_cols):
    for c in tme_cols:
        gby_cols = c.split('_')[:-2]
        gp = train_df[gby_cols + [c]].groupby(gby_cols).mean().reset_index()
        df = df.merge(gp, on=gby_cols, how='left')
        df.fillna(df[c].median(), inplace=True)
    return df

def train_cats(df):
    for n, c in df.items():
        if is_object_dtype(c):
            df[n] = c.astype('category').cat.as_ordered()

def apply_cats(df, train_df):
    for n,c in df.items():
        if (n in train_df.columns) and (train_df[n].dtype.name=='category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n].cat.set_categories(train_df[n].cat.categories, ordered=True, inplace=True)

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


# ## Ranking


# In[11]:


def rmse(pred,y):
    return round(math.sqrt(((pred-y)**2).mean()), 4)

def m_rmse(m, xs, y):
    return rmse(m.predict(xs), y)

def eval_model(m, trn_xs, trn_y, val_xs, val_y):
    return m_rmse(m, trn_xs, trn_y), m_rmse(m, val_xs, val_y)

def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))

def cluster_columns(df, figsize=(10,6), font_size=12, fig_path=img_path/'cc.png'):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns.tolist(), orientation='left', leaf_font_size=font_size)
    plt.show()
    plt.savefig(fig_path)
    
def rf(xs, y, n_estimators=40, max_features=0.5, min_samples_leaf=25, max_samples=500_000, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  max_samples=max_samples, **kwargs).fit(xs, y)

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def plot_fi(fi, fig=img_path/'fi.png'):
    fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
    plt.savefig(fig)


# ## Final

# In[61]:


#def final_rf(xs, y, n_estimators=120, max_features=0.5, min_samples_leaf=5, **kwargs):
#    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
#                                  min_samples_leaf=min_samples_leaf, max_samples=2_000_000, **kwargs).fit(xs, y)

def final_rf(xs, y, n_estimators=120, max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, **kwargs).fit(xs, y)


# In[59]:


print("Loading model ...")
model = pickle.load((model_path/'final_rf_model.pkl').open('rb'))
final_cols = pickle.load((model_path/'final_columns.pkl').open('rb'))


# ## Testing

# In[64]:


def mae(p, t): return np.mean(abs(p - t))


def testing(path, n_pct=0.1, topN=5):
    fast_bench_csv = list(path.glob('fast_bench/2_BenchmarkData/*.csv'))
    origin_csv = list(path.glob('origin/2_BenchmarkData/*.csv'))

    for f, o in zip(fast_bench_csv, origin_csv):
        fast_bench_df = pd.read_csv(f, low_memory=False)
        origin_df = pd.read_csv(o, low_memory=False)

        cols = fast_bench_df.columns
        assert (cols == origin_df.columns).all() and fast_bench_df.shape == origin_df.shape

        fast_bench_df.replace(" " , '0', inplace=True)
        for n, c in fast_bench_df[cols[10:]].items():
            if is_object_dtype(c):
                fast_bench_df[n] = pd.to_numeric(c).astype('float')

#        import pdb; pdb.set_trace()
#        for n, c in fast_bench_df[cols[10:]].items():
#            val = []
#            for o in c.values:
#                if o == '':
#                    val.append(0)
#                else:
#                    val.append(
#            fast_bench_df[n] = np.array([eval(o) for o in c.values])
        gflops_preds = fast_bench_df[cols[10:]].values
        n, m = gflops_preds.shape
        num_preds = int(m * n_pct)
        topN_preds = np.argsort(-gflops_preds)[:, :num_preds]

        gflops = origin_df[cols[10:]].values
        topN_target = np.argsort(-gflops)[:, :topN]

        #import pdb; pdb.set_trace()

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


# In[65]:


print("Testing model ...")


# In[66]:


testing(nn_path, n_pct=0.05)
testing(nt_path, n_pct=0.05)
testing(tn_path, n_pct=0.15)

