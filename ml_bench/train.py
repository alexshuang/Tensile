#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_bool_dtype, is_numeric_dtype, is_object_dtype, is_categorical_dtype, is_integer_dtype, is_float_dtype
from fastai.fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
#from IPython.display import Image, display_svg, SVG
#from dtreeviz.trees import *
#from sklearn.tree import export_graphviz
#import scipy
#from scipy.cluster import hierarchy as hc
#from sklearn.inspection import plot_partial_dependence
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
parser.description = "train and testing"
parser.add_argument("--data_dir", type=str)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--train_dir_name", type=str, default='')
parser.add_argument("--test_dir_name", type=str, default='')
args = parser.parse_args()

path = Path(args.data_dir)
train_path = path
#test_path = path/args.test_dir_name
img_path = path/'imgs'
model_path = path/'models'
img_path.mkdir(exist_ok=True)
model_path.mkdir(exist_ok=True)


# In[3]:


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

def train_cats(df, cat_cols=None):
    for n, c in df.items():
        if is_object_dtype(c) or (cat_cols is not None and n in cat_cols):
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

def final_rf(xs, y, n_estimators=200, max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, **kwargs).fit(xs, y)


if not (model_path/'final_rf_model.pkl').is_file():
    print("Train RF model ...")
    final_cols = ['AreaC', 'NumElementsPerThread',
            'TotalFlops', 'LdsNumElements', 'StoreRemapVectorWidth', 'SizeL',
            'KernelName', 'MacroTile1', 'WorkGroup_1', 'WorkGroup_0',
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
    dep_var = 'Ranking'

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
    y, valid_y = np.log1p(train_df[dep_var].values), np.log1p(valid_df[dep_var].values)
    xs = train_df.drop(dep_var, axis=1)
    valid_xs = valid_df.drop(dep_var, axis=1)
    xs_final, valid_xs_final = xs[final_cols], valid_xs[final_cols]

    rf_m = final_rf(xs_final, y)
    print(eval_model(rf_m, xs_final, y, valid_xs_final, valid_y))

    pickle.dump(xs_final.columns.values, (model_path/'final_columns.pkl').open('wb'))
    pickle.dump(rf_m, (model_path/'final_rf_model.pkl').open('wb'))


# ## NN


def rmse(pred,y):
    return torch.sqrt(((pred-y)**2).mean())

def get_learn(to_nn):
    dls = to_nn.dataloaders(1024)
    learn = tabular_learner(dls, y_range=(0, 1), layers=[750, 375], n_out=1, loss_func=F.mse_loss, metrics=[rmse])
    return learn


if not ((model_path/'to_nn.pkl').is_file() and ):
    print("Train NN model ...")
    train_df = pd.read_feather(train_path/'train.feat')
    valid_df = pd.read_feather(train_path/'valid.feat')
    train_df = train_df[train_df['GFlops'] > 0].reset_index(drop=True)
    valid_df = valid_df[valid_df['GFlops'] > 0].reset_index(drop=True)
    df_nn_final = pd.concat([train_df, valid_df], ignore_index=True)
    preproc_df(df_nn_final)
    df_nn_final = df_nn_final[final_cols + [dep_var]]
    df_nn_final[dep_var] = np.log1p(df_nn_final[dep_var])
    cont_var = ['AreaC', 'TotalFlops', 'SizeL', 'LDB', 'AspectRatioA', 'SizeK',
              'AspectRatioC', 'LDA']
    cat_var = list(set(final_cols) - set(cont_var))
    procs_nn = [Categorify, Normalize]
    idxs = np.arange(len(df_nn_final))
    splits = (list(idxs[:len(train_df)]),list(idxs[len(train_df):]))
    to_nn = TabularPandas(df_nn_final, procs_nn, cat_var, cont_var, splits=splits, y_names=dep_var)
    pickle.dump(to_nn, (model_path/'to_nn.pkl').open('wb'))

    learn = get_learn(to_nn)
    learn.lr_find()
    plt.savefig(img_path/'lr_find.png')
    learn.fit_one_cycle(3, 3e-3)
    learn.save('nn_m');


# ## Testing

# In[5]:


def mae(p, t): return np.mean(abs(p - t))

def mee(p, t, eff_err): return np.mean(np.abs(p - t) < t * eff_err)

def testing(test_csv, dep_var='Ranking', n_pct=0.1, topN=5, eff_err=0.015):
#    # rf
#    print("Loading RF model ...")
#    rf_m = pickle.load((model_path/'final_rf_model.pkl').open('rb'))
#    final_cols = pickle.load((model_path/'final_columns.pkl').open('rb'))
#    # nn
#    print("Loading NN model ...")
#    to_nn = pickle.load((model_path/'to_nn.pkl').open('rb'))
#    learn = get_learn(to_nn)
#    learn.load('nn_m');
    
    for tf in test_csv:
        print(f"{tf} ...")
        num_solution = re.findall(ns_pat, tf.stem)[0]
        assert num_solution.isdecimal()
        num_solution = eval(num_solution)

        #df = pd.read_feather(tf)
        df = pd.read_csv(tf, low_memory=False)
        gflops = df['GFlops'].values
        gflops = gflops.reshape(-1, num_solution)
        n = gflops.shape[0]
        topN_target = np.argsort(-gflops)[:, :topN]

        preproc_df(df)
        
        # rf
        rf_df = df[final_cols].copy()
        train_cats(rf_df)
        categorify(rf_df)
        rf_preds = rf_m.predict(rf_df)
        rf_preds = np.expm1(rf_preds)
        
        # nn
        nn_df = df[final_cols.tolist() + [dep_var]].copy()
        cont_var = ['AreaC', 'TotalFlops', 'SizeL', 'LDB', 'AspectRatioA', 'SizeK',
          'AspectRatioC', 'LDA']
        cat_var = list(set(final_cols) - set(cont_var))
        dl = learn.dls.test_dl(nn_df)
        nn_preds = learn.get_preds(dl=dl)
        nn_preds = np.expm1(nn_preds[0])
        nn_preds = nn_preds.reshape(-1)
        
        # mean
        preds = np.stack([rf_preds, nn_preds])
        preds = preds.mean(axis=0)
        
        preds = preds.reshape(gflops.shape)
        num_preds = int(preds.shape[1] * n_pct)
        topN_preds = np.argsort(preds)[:, :num_preds]

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

        print(f"{tf.stem}: {n_pct*100}%/{num_preds} solutions, top1 accuracy: {np.sum(top1_acc)/n*100:.2f}%, top{topN} accuracy: {np.sum(acc)/n*100:.2f}%")
        print(f"\t\tefficiency error < 1.5%: {mee(gflops_preds, gflops_target, eff_err) * 100:.2f}%")
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        x_axis = np.arange(n)
        j = n // 9
        for i, ax in enumerate(axes.flatten()):
            ax.plot(x_axis[i*j:i*j+j], gflops_preds[i*j:i*j+j], label='preds')
            ax.plot(x_axis[i*j:i*j+j], gflops_target[i*j:i*j+j], label='target')
        plt.subplots_adjust(right=1.5)
        plt.legend()
        plt.show()


# In[6]:


print("validating ...")
#test_csv = list(train_path.glob('**/valid_N*.feat'))
test_csv = list(train_path.glob('**/valid_N*.csv'))
ns_pat = re.compile(r'_N(.*)')
print(f"{test_csv}")

# rf
print("Loading RF model ...")
rf_m = pickle.load((model_path/'final_rf_model.pkl').open('rb'))
final_cols = pickle.load((model_path/'final_columns.pkl').open('rb'))
# nn
print("Loading NN model ...")
to_nn = pickle.load((model_path/'to_nn.pkl').open('rb'))
learn = get_learn(to_nn)
learn.load('nn_m');


# In[10]:


testing(test_csv, n_pct=0.02)


# In[9]:


testing(test_csv, n_pct=0.1)


# In[7]:


testing(test_csv, n_pct=0.15)


# In[8]:


testing(test_csv, n_pct=0.2)

