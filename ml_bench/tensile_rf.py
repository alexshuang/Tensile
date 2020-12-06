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
from IPython.display import Image, display_svg, SVG
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


path = Path('data/fast_bench/inc1')
nn_path = path/'nn'
tn_path = path/'tn'
nt_path = path/'nt'
model_path = Path('data/inc1/models')
img_path = Path('data/fast_bench/inc1/imgs')
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


if not (model_path/'final_rf_model.pkl').is_file():
            #'SolutionName', 'MacroTile1', 'WorkGroup_1', 'WorkGroup_0',
            #'MacroTile1', 'WorkGroup_1', 'WorkGroup_0',
    final_cols = ['AreaC', 'NumElementsPerThread',
            'TotalFlops', 'LdsNumElements', 'StoreRemapVectorWidth', 'SizeL',
            'SolutionName', 'MacroTile1', 'WorkGroup_1', 'WorkGroup_0',
            'MacroTile0', 'LDB', 'AspectRatioA', 'LdsOffsetB_Blk', 'LdsOffsetA_Blk',
            'DepthU', 'LdsNumElementsAlignedB', 'SizeK',
            'AspectRatioC', 'LSCA', 'LoopIters', 'AssertFree0ElementMultiple',
            'ThreadTile0', 'LDA', 'LVPA', 'LSCB',
            'ThreadTile_1', 'LdsOffsetB', 'MatrixInstK',
            'LVPB', 'MIWaveGroup_0', 'GuaranteeNoPartialA', 'SubGroup0',
            'MatrixInstruction_1', 'ThreadTile1', 
            'GlobalLoadVectorWidthB', 'SubGroup1', 'MIWaveGroup_1',
            '_UseSgprForGRO', 'NumLoadsPerpendicularB', 'NumLoadsB',
            'GlobalLoadVectorWidthA', 'GlobalReadVectorWidth', 'ThreadTile_0',
            'LSPB', 'LSPA', 'LVCA', 'LVCB',
            'WorkGroupMapping']

    train_df = pd.read_feather(train_path/'train.feat')
    valid_df = pd.read_feather(train_path/'valid.feat')
    train_df = train_df[train_df['GFlops'] > 0].reset_index(drop=True)
    valid_df = valid_df[valid_df['GFlops'] > 0].reset_index(drop=True)

    preproc_df(train_df)
    preproc_df(valid_df)
    train_cats(train_df)
    apply_cats(valid_df, train_df)
    categorify(train_df)
    categorify(valid_df)

    train_df.drop(['GFlops'], axis=1, inplace=True)
    valid_df.drop(['GFlops'], axis=1, inplace=True)
    dep_var = 'Ranking'
    y, valid_y = np.log1p(train_df[dep_var].values), np.log1p(valid_df[dep_var].values)
    xs = train_df.drop(dep_var, axis=1)
    valid_xs = valid_df.drop(dep_var, axis=1)
    xs_final, valid_xs_final = xs[final_cols].copy(), valid_xs[final_cols].copy()
    del xs, valid_xs

    cluster_columns(valid_xs_final, figsize=(12, 12), font_size=9);

    model = final_rf(xs_final, y)
    print("Final", eval_model(model, xs_final, y, valid_xs_final, valid_y))

    fi = rf_feat_importance(model, xs_final)
    plot_fi(fi[:30])

    pickle.dump(xs_final.columns.values, (model_path/'final_columns.pkl').open('wb'))
    del xs_final, y, valid_xs_final, valid_y
    pickle.dump(model, (model_path/'final_rf_model.pkl').open('wb'))
else:
    model = pickle.load((model_path/'final_rf_model.pkl').open('rb'))
    final_cols = pickle.load((model_path/'final_columns.pkl').open('rb'))


# ## Testing

# In[64]:


def mae(p, t): return np.mean(abs(p - t))


def testing(origin_csv, fast_bench_csv, n_pct=0.05, topN=5):
    df = pd.read_csv(fast_bench_csv, low_memory=False)
    cols = df.columns.values[10:]
    for n in cols:
        data = []
        for o in df[n].values:
            try:
                val = eval(o)
            except:
                val = 0
            data.append(val) 
        df[n] = np.array(data)
    preds = df[cols].values
    n = preds.shape[0]
    gflops_preds = np.sort(preds)[:, -1].reshape(-1)

    origin_df = pd.read_csv(origin_csv, low_memory=False)
    gflops = origin_df['GFlops'].values.reshape(n, -1)
    gflops_target = np.sort(gflops)[:, -1].reshape(-1)

    import pdb; pdb.set_trace()
    print(f"{origin_csv.parent.parent.stem}: mean errors: {mae(gflops_preds, gflops_target):.2f} GFlops")
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    x_axis = np.arange(n)
    j = n // 9
    for i, ax in enumerate(axes.flatten()):
        ax.plot(x_axis[i*j:i*j+j], gflops_preds[i*j:i*j+j], label='prediction')
        ax.plot(x_axis[i*j:i*j+j], gflops_target[i*j:i*j+j], label='target')
    plt.subplots_adjust(right=1.5)
    plt.legend()
    plt.show()
    plt.gcf().savefig(img_path/f'{origin_csv.parent.parent.stem}_pct{n_pct}_top{topN}.png', dpi=600, bbox_inches='tight')


testing(nn_path/'origin/Cijk_Ailk_Bljk_HBH_00.csv', nn_path/'fast_bench/Cijk_Ailk_Bljk_HBH_00.csv', n_pct=0.05)
#testing(tn_path/'origin/Cijk_Alik_Bljk_HBH_00.csv', tn_path/'fast_bench/00_Final.csv', n_pct=0.1)
#testing(nt_path/'origin/00_Final.csv', nt_path/'fast_bench/00_Final.csv', n_pct=0.05)

