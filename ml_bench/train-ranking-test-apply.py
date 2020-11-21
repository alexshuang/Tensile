#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_bool_dtype, is_numeric_dtype, is_object_dtype, is_categorical_dtype, is_integer_dtype, is_float_dtype
#from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
#from IPython.display import Image, display_svg, SVG
#from dtreeviz.trees import *
from sklearn.tree import export_graphviz
#from scipy.cluster import hierarchy as hc
from sklearn.inspection import plot_partial_dependence
import yaml
import os
import math
import random
import pickle
import time
from tqdm import tqdm
from collections import defaultdict
from sklearn import base
from sklearn.model_selection import KFold
from pathlib import Path


# In[3]:


path = Path('data/')


# In[5]:


train_df = pd.read_feather(path/'train/train_and_valid_raw_full.feat')
valid_df = pd.read_feather(path/'test/inc1_raw_full.feat')


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
        elif is_categorical_dtype(c):
            df[n] = pd.Categorical(c).codes + 1

def preproc_df(df):
    #df['PadA'] = df['LDA'] - df['SizeL'] if df['PT_TransposeA'] else df['LDA'] - df['SizeI']
    #df['PadB'] = df['LDB'] - df['SizeJ'] if df['PT_TransposeB'] else df['LDB'] - df['SizeL']
    df['PadA'] = df['LDA'] - df['SizeI']
    df['PadB'] = df['LDB'] - df['SizeL']
    df['PadC'] = df['LDC'] - df['SizeI']
    df['AspectRatioA'] = df['SizeL'] / df['SizeI']
    df['AspectRatioB'] = df['SizeJ'] / df['SizeL']
    df['AspectRatioC'] = df['SizeJ'] / df['SizeI']
    df['AreaA'] = (df['SizeI'] * df['SizeL']).astype('int64')
    df['AreaB'] = (df['SizeJ'] * df['SizeL']).astype('int64')
    df['AreaC'] = (df['SizeI'] * df['SizeJ']).astype('int64')
    df['AoverB'] = df['AreaA'] / df['AreaB']


# In[12]:


train_df = train_df[train_df['GFlops'] > 0].reset_index(drop=True)
valid_df = valid_df[valid_df['GFlops'] > 0].reset_index(drop=True)
preproc_df(train_df)
preproc_df(valid_df)


# In[ ]:


problem_size_cols = ['AreaA', 'AreaB', 'AreaC', 'SizeI', 'SizeJ', 'SizeK', 'SizeL',
                    'LDA', 'LDB', 'LDC', 'LDD', 'TotalFlops', 'AspectRatioA',
                    'AspectRatioB', 'AspectRatioC', 'AoverB', 'PadA', 'PadB', 'PadC']
#final_cols = ['AreaC', 'TotalFlops', 'LdsNumElements', 'SolutionName',
final_cols = ['AreaC', 'TotalFlops', 'LdsNumElements',
       'NumElementsPerThread', 'SizeL',
       'AreaB', 'AspectRatioB', 'SizeJ', 'StoreRemapVectorWidth', 'AreaA',
       'AspectRatioA', 'LdsOffsetB_Blk', 'LDB', 'LdsNumElementsAlignedB',
       'LoopUnroll', 'AspectRatioC', 'PadB', 'AoverB', 'SizeK',
       'LdsOffsetA_Blk', 'MacroTile1', 'LDC', 'LSCA', 'PadA', 'LSCB', 'SizeI',
       'LDA', 'LoopIters', 'ThreadTile_1', 'GlobalReadVectorWidth',
       'LdsOffsetB', 'NumLoadsB',
       'NumLoadsPerpendicularA']
dep_var = 'Target'
train_df[dep_var] = train_df.Ranking < 0.02
valid_df[dep_var] = valid_df.Ranking < 0.02

y, valid_y = train_df[dep_var].values.astype('int'), valid_df[dep_var].values.astype('int')
xs_final = train_df[final_cols].copy()
del train_df
valid_xs_final = valid_df[final_cols].copy()
del valid_df


#def add_solution_name(df):
#    snames = defaultdict(lambda: [])
#    for n, c in df.items():


# In[9]:


def acc(pred, target):
    return (pred == target).mean()

def m_acc(m, xs, y):
    return acc(m.predict(xs), y)

def eval_model(m, trn_xs, trn_y, val_xs, val_y):
    return m_acc(m, trn_xs, trn_y), m_acc(m, val_xs, val_y)

def rf(xs, y, n_estimators=40, max_features=0.5, min_samples_leaf=25, **kwargs):
    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  max_samples=200_000, oob_score=True, **kwargs).fit(xs, y)

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

def tolist(x):
    if isinstance(x, str): return [x]
    elif isinstance(x, tuple): return list(x)
    return x

def tostr(x):
    if isinstance(x, [list, tuple]): return '_'.join(x)
    return x

def kfold_target_encoding(train_df, recipies, n_fold=5, drop=True, tail_name='target'):
    tme_cols = []
    train_new = train_df.copy()
    for i, (gby_col, target_col, op) in enumerate(recipies):
        kf = KFold(n_splits=n_fold, shuffle=True)#, random_state=21)
        for tr_ind, val_ind in kf.split(train_df):
            trn_df, val_df = train_df.iloc[tr_ind], train_df.iloc[val_ind]
            agg = trn_df.groupby(gby_col)[target_col].agg(op)
            col_names = ['_'.join([gby_col, c, tail_name]) for c in agg.columns]
            agg.columns = col_names
            target_mean_enc = agg.reset_index().copy()
            for c in col_names:
                train_new.loc[val_ind, c] = val_df[gby_col].map(agg[c])
        train_new.fillna(train_df[target_col].median(), inplace=True)
    return train_new, target_mean_enc

def gen_target_mean_enc(trn_df, recipies):
    return kfold_target_encoding(trn_df, recipies)

# apply target mean encoding by train dataframe
def apply_target_mean_enc(df, tme, drop=True):
    gby_col = tme.columns[0]
    df = df.merge(tme, on=gby_col, how='left')
    for n in tme.columns[1:]:
        df.fillna(df[n].median(), inplace=True)
    return df

def param_bench(model, params, trn_xs, trn_y, val_xs, val_y):
    res = []
    for f in params['max_features']:
        for s in params['min_samples_leaf']:
            m = model(trn_xs, trn_y, max_features=f, min_samples_leaf=s)
            res.append((f'max_features={f}, min_samples_leaf={s}',
                        m_acc(m, trn_xs, trn_y), m_acc(m, val_xs, val_y)))
            del m
    res_sorted = sorted(res, key=lambda x: x[2])
    return res_sorted


# In[ ]:


#enc_cols = ['SolutionName']
#xs_final['Target'] = y
#agg_op = ['mean']
#dep_var = 'Target'
#recipies = [(c, dep_var, agg_op) for c in enc_cols]
#recipies


# In[ ]:


train_new = xs_final.copy()
valid_new = valid_xs_final.copy()

#train_new, tme = gen_target_mean_enc(xs_final, recipies)
# xs_final.drop(['Target', 'SolutionName'], axis=1, inplace=True)
#train_new.drop(['Target'], axis=1, inplace=True)
#train_new.drop(['SolutionName'], axis=1, inplace=True)

#valid_new = apply_target_mean_enc(valid_xs_final, tme)
#valid_new.drop(['SolutionName'], axis=1, inplace=True)

#import pdb; pdb.set_trace()
#
#train_cats(train_new)
#apply_cats(valid_new, train_new)
#categorify(train_new)
#categorify(valid_new)

#import pdb; pdb.set_trace()
model = rf(train_new, y)
print("RF", eval_model(model, train_new, y, valid_new, valid_y))


# In[26]:


print("Saving Final Dataset ...")
pickle.dump(train_new, open(str(path/'xs_final.pkl'), 'wb'))
pickle.dump(valid_new, open(str(path/'valid_xs_final.pkl'), 'wb'))
pickle.dump(y, open(str(path/'y.pkl'), 'wb'))
pickle.dump(valid_y, open(str(path/'valid_y.pkl'), 'wb'))
#pickle.dump(tme, open(str(path/'target_mean_enc.pkl'), 'wb'))
pickle.dump(train_new.columns, open(str(path/'columns_final.pkl'), 'wb'))


# ## Final

# In[4]:


xs_final = pickle.load((path/'xs_final.pkl').open('rb'))
valid_xs_final = pickle.load((path/'valid_xs_final.pkl').open('rb'))
y = pickle.load((path/'y.pkl').open('rb'))
valid_y = pickle.load((path/'valid_y.pkl').open('rb'))


# In[ ]:


params = {
    'max_features': [0.5],
    'min_samples_leaf': [10, 15],
}

#print("Benchmarking RF parameters ...")
#res = param_bench(rf, params, train_new, y, valid_new, valid_y)
#for o in res:
#    print(f"{o[0]}: train = {o[1]:.4f}, valid = {o[2]:.4f}")


# In[ ]:


def final_rf(xs, y, n_estimators=160, max_features=0.5, min_samples_leaf=10, **kwargs):
    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, max_samples=1_000_000, **kwargs).fit(xs, y)


# In[ ]:


print("Train final RF model ...")
model = final_rf(train_new, y)
print("Final RF", eval_model(model, train_new, y, valid_new, valid_y))


# In[ ]:


pickle.dump(train_new, (path/'rf_model_final.pkl').open('wb'))


# In[ ]:



