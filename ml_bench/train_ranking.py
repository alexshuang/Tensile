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
from scipy.cluster import hierarchy as hc
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


# In[2]:


#torch.cuda.is_available()


# ## Look at Data

# In[3]:


path = Path('data/train/')


# In[4]:


#path.ls()


# In[5]:


train_df = pd.read_feather(path/'train_raw_full.feat')
valid_df = pd.read_feather(path/'test_raw_full.feat')


pd.options.display.max_rows = None


# In[8]:


print(train_df.describe().T)


# In[10]:


for n, c in train_df.items():
    print(f"{n}: {c.unique()}")


# In[9]:


train_df[train_df.GFlops <= 0].GFlops.value_counts(), valid_df[valid_df.GFlops <= 0].GFlops.value_counts()


# In[10]:


((train_df.GFlops <= 0).sum() / len(train_df) * 100), ((valid_df.GFlops <= 0).sum() / len(valid_df) * 100)


# In[6]:


train_df = train_df[train_df['GFlops'] > 0].reset_index(drop=True)
valid_df = valid_df[valid_df['GFlops'] > 0].reset_index(drop=True)
len(train_df), len(valid_df)


# In[7]:


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

def preproc_df(df):
    df['PadA'] = df['LDA'] - df['SizeI']
    df['PadB'] = df['LDB'] - df['SizeL']
    df['PadC'] = df['LDC'] - df['SizeI']
    df['AspectRatioA'] = df['SizeL'] / df['SizeI']
    df['AspectRatioB'] = df['SizeJ'] / df['SizeL']
    df['AspectRatioC'] = df['SizeJ'] / df['SizeI']
    df['AreaA'] = df['SizeI'] * df['SizeL']
    df['AreaB'] = df['SizeJ'] * df['SizeL']
    df['AreaC'] = df['SizeI'] * df['SizeJ']
    df['AoverB'] = df['AreaA'] / df['AreaB']
#     df['TotalFlops'] = df['TotalFlops'] / 1e9
    dup_cols = ['LDD', 'MacroTileA', 'MacroTileB','SubGroupA', 'SubGroupB',
                'ThreadTileA', 'ThreadTileB', 'MatrixInstBM', 'MatrixInstN']
    df.drop(dup_cols, axis=1, inplace=True)


# In[8]:


preproc_df(train_df)


# In[9]:


preproc_df(valid_df)


# In[19]:


# df_train, tme_cols = gen_tme_feat(train_df)
# df_valid = apply_tme_feat(valid_df, df_train, tme_cols)
# df = pd.concat([df_train, df_valid], ignore_index=True)
# num_train = len(df_train)
# del train_df, valid_df, df_train, df_valid


# In[10]:


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


# In[11]:


train_cats(train_df)
apply_cats(valid_df, train_df)
categorify(train_df)
categorify(valid_df)


# In[19]:


train_df.head().T


# ## Ranking

# In[12]:


train_df['Target'] = train_df.Ranking < 0.1
valid_df['Target'] = valid_df.Ranking < 0.1


# In[13]:


train_df.drop(['GFlops', 'Ranking'], axis=1, inplace=True)
valid_df.drop(['GFlops', 'Ranking'], axis=1, inplace=True)


# In[14]:


dep_var = 'Target'
y, valid_y = train_df[dep_var].values.astype('int32'), valid_df[dep_var].values.astype('int32')
xs = train_df.drop(dep_var, axis=1)
valid_xs = valid_df.drop(dep_var, axis=1)


# In[15]:


def acc(pred, target):
    return (pred == target).mean()

def m_acc(m, xs, y):
    return acc(m.predict(xs), y)

def eval_model(m, trn_xs=xs, trn_y=y, val_xs=valid_xs, val_y=valid_y):
    return print(m_acc(m, trn_xs, trn_y), m_acc(m, val_xs, val_y))

def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))

def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns.tolist(), orientation='left', leaf_font_size=font_size)
    plt.show()


# In[27]:


#get_ipython().run_cell_magic('time', '', 't = DecisionTreeRegressor()\nt.fit(xs, y)\nm_acc(t, xs, y), m_acc(t, valid_xs, valid_y)')


# In[26]:


#t.get_n_leaves(), len(xs)


# In[30]:


#t = DecisionTreeRegressor(min_samples_leaf=25)
#t.fit(xs, y)
#m_rmse(t, xs, y), m_rmse(t, valid_xs, valid_y)


# In[31]:


#t.get_n_leaves(), len(xs)


# In[16]:


def rf(xs, y, n_estimators=40, max_features=0.5, min_samples_leaf=25, **kwargs):
    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  max_samples=200_000, oob_score=True, **kwargs).fit(xs, y)


# In[17]:


m = rf(xs, y)
eval_model(m)


# In[18]:


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[19]:


fi = rf_feat_importance(m, xs)
#plot_fi(fi[:30])


# In[20]:


to_keep = fi[fi.imp > 0.005].cols
xs_keep, valid_xs_keep = xs[to_keep], valid_xs[to_keep]
m = rf(xs_keep, y)
eval_model(m, xs_keep, y, valid_xs_keep, valid_y)


# In[21]:


len(xs_keep.columns), xs_keep.columns


# In[22]:


cluster_columns(xs_keep, figsize=(12, 12), font_size=9)


import pdb; pdb.set_trace()

# In[ ]:


dup_cols = ['NumElementsPerThread', 'NumGlobalWriteVectorsPerThread',
            'LdsOffsetB', 'LdsNumElementsAlignedA',
            'LoopUnroll', 'DepthU',
            'ThreadTile1', 'MIWaveTile_1',
           'MatrixInstruction_2', 'MatrixInstK']

def get_oob(x, y):
    m = RandomForestClassifier(n_jobs=-1, n_estimators=40, max_features=0.5,
                                  min_samples_leaf=15,
                                  max_samples=100_000, oob_score=True).fit(xs, y)
    return m.oob_score_

print(f"original: {get_oob(xs_keep, y)}")
res = {c:get_oob(xs_keep.drop(c, axis=1), y) for c in dup_cols if c in xs_keep}
for k, v in res.items():
    print(f"{k}: {v}")

# for c in dup_cols:
#     m = oobs = {c:get_oob(xs.drop(c, axis=1), y) for c in cols}(xs_keep.drop(c, axis=1), y)
#     print(c, eval_model(m, xs_keep.drop(c, axis=1), y, valid_xs_keep.drop(c, axis=1), valid_y))
#     del m


# In[24]:


get_ipython().run_cell_magic('time', '', "drop_cols = ['MatrixInstruction_2', 'NumGlobalWriteVectorsPerThread', 'LdsNumElementsAlignedA', 'DepthU', 'MIWaveTile_1']\nfor c in drop_cols:\n    if c in xs_keep and c in valid_xs_keep:\n        xs_keep2 = xs_keep.drop(c, axis=1)\n        valid_xs_keep2 = valid_xs_keep.drop(c, axis=1)\n\ndel m\nm = rf(xs_keep2, y)\neval_model(m, xs_keep2, y, valid_xs_keep2, valid_y)")


# In[25]:


fi = rf_feat_importance(m, xs_keep2)
plot_fi(fi)


# In[26]:


(path/'xs_final.pkl').save(xs_keep2)
(path/'valid_xs_final.pkl').save(valid_xs_keep2)
(path/'y.pkl').save(y)
(path/'valid_y.pkl').save(valid_y)


# ## Final

# In[28]:


def final_rf(xs, y, n_estimators=100, max_features=0.5, min_samples_leaf=15, **kwargs):
    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, oob_score=True, **kwargs).fit(xs, y)


# In[29]:


model = final_rf(xs_keep2, y)
eval_model(model, xs_keep2, y, valid_xs_keep2, valid_y)


# ## NN

# In[4]:


xs_final = (path/'xs_final.pkl').load()
valid_xs_final = (path/'valid_xs_final.pkl').load()
y = (path/'y.pkl').load()
valid_y = (path/'valid_y.pkl').load()


# In[5]:


df = pd.concat([xs_final, valid_xs_final], ignore_index=True)
df['GFlops'] = np.concatenate([y, valid_y])
splits = (list(np.arange(0, len(xs_final))), list(np.arange(len(valid_xs_final), len(df))))


# In[6]:


del xs_final, valid_xs_final, y, valid_y


# In[7]:


df.columns


# In[9]:


cat_cols = ['SolutionName']
cont_cols = list(set(df.columns) - set(cat_cols))


# In[10]:


dep_var = 'GFlops'
procs_nn = [Categorify, Normalize]
to_nn = TabularPandas(df, procs_nn, cat_cols, cont_cols, splits=splits, y_names=dep_var)
# to_nn = TabularPandas(df, procs_nn, df.columns, [],           
# (path/f'to_nn.pkl').save(to_nn)


# In[ ]:


dls = to_nn.dataloaders(256)


# In[ ]:


df.GFlops.max(), df.GFlops.min()


# In[150]:


learn = tabular_learner(dls, path=path, y_range=[3, 11], layers=[500, 250], n_out=1,
                        loss_func=F.mse_loss, metrics=rmse)


# In[144]:


learn.lr_find()


# In[151]:


learn.fit_one_cycle(7, 1e-3)


# In[ ]:





# In[67]:


cols = list(fi[fi.imp < 0.005].cols)
cnt = 10
res = [(c, np.mean([remove_redundant_feat(c) for i in range(cnt)])) for c in cols]
res_sorted = sorted(res, key=lambda x: x[1])


# In[109]:





# In[110]:


dup_cols = ['LdsBlockSizePerPadB', 'GlobalReadVectorWidth']
xs.drop(dup_cols, axis=1, inplace=True)
valid_xs.drop(dup_cols, axis=1, inplace=True)


# In[75]:


for k, v in res_sorted:
    print(f"{k}: {v}")


# In[ ]:





# In[ ]:





# In[77]:


dup_cols = ['LdsBlockSizePerPadB', 'GlobalReadVectorWidth']
xs.drop(dup_cols, axis=1, inplace=True)
valid_xs.drop(dup_cols, axis=1, inplace=True)


# In[82]:


m = final_rf(xs, y)
eval_model(m)


# In[83]:


fi = rf_feat_importance(m, xs)
plot_fi(fi[:30])


# In[91]:


keep_idxs = fi[fi.imp > 0.0001].cols
xs_keep, valid_xs_keep = xs[keep_idxs].copy(), valid_xs[keep_idxs].copy()
len(xs_keep)


# In[98]:


def eval_model(x=xs, x2=valid_xs, cnt=10):
    r = []
    for i in range(cnt):
        m = final_rf(x, y)
        r.append(m_rmse(m, x2, valid_y))
        del m
    return np.mean(r)


# In[99]:


eval_model(xs_keep, valid_xs_keep)


# In[100]:


eval_model()


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


path = Path('data/nn_tiny_profile/')


# In[4]:


path.ls()


# In[34]:


get_ipython().run_cell_magic('time', '', 'for f in list(path.glob("**/*.yaml")):\n    dataset_create(f.with_suffix(\'\'))')


# In[2]:


def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))

def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()


# ## Prepare Data

# In[3]:


df = pd.read_csv("data/src/rocblas_hpa_hgemm_nn_inc1_asm_full/Cijk_Ailk_Bljk_HBH_00.csv", low_memory=False)


# In[23]:


tot_cols = []
for f in list(path.glob("**/*.csv")):
    print(f"loading {f} ...")
    df = pd.read_csv(f, low_memory=False)
    cols = df.columns[10:]
    tot_cols.extend(cols)
tot_cols = list(set(tot_cols))


# In[25]:


cols = np.unique(['_'.join(o.split('_')[3:]) for o in tot_cols])


# In[27]:


len(cols), cols[:3]


# In[19]:


feat = yaml.safe_load(open("data/src/rocblas_hpa_hgemm_nn_inc1_asm_full/Cijk_Ailk_Bljk_HBH_00.yaml"))


# In[20]:


df.head()


# In[21]:


feat[3]


# In[11]:


def dataset_create(path:pathlib):
    for f in list(path.glob("**/*.csv")):
        print(f"loading {f} ...")
        df = pd.read_csv(f, low_memory=False)
        feat_f = f.with_suffix('.yaml')
        print(f"loading {feat_f} ...")
        feat = yaml.safe_load(feat_f.open())


# In[ ]:





# In[5]:


to = (path/'rf_to.pkl').load()
xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y
y, valid_y = np.log1p(y), np.log1p(valid_y)


# ## Interp

# In[4]:


xs = (path/'train_xs.pkl').load()
y = (path/'train_y.pkl').load()
valid_xs = (path/'valid_xs.pkl').load()
valid_y = (path/'valid_y.pkl').load()


# In[10]:


t = DecisionTreeRegressor(max_leaf_nodes=4)
t.fit(xs, y);


# In[11]:


draw_tree(t, xs, size=7, leaves_parallel=True, precision=2)


# In[13]:


samp_idx = np.random.permutation(len(y))[:500]
dtreeviz(t, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, 'GFlops',
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')


# In[14]:


def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


# In[16]:


m_rmse(t, xs, y), m_rmse(t, valid_xs, valid_y)


# In[17]:


xs.columns


# In[18]:


def rf(xs, y, n_estimators=40, max_samples=100000,
       max_features=0.5, min_samples_leaf=30, **kwargs):
    return RandomForestRegressor(n_jobs=10, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


# In[19]:


get_ipython().run_cell_magic('time', '', 'm = rf(xs, y, min_samples_leaf=20)e\nm_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)')


# In[20]:


get_ipython().run_cell_magic('time', '', 'm = rf(xs, y, min_samples_leaf=10)\nm_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)')


# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


(path/'train_xs.pkl').save(xs)
(path/'train_y.pkl').save(y)
(path/'valid_xs.pkl').save(valid_xs)
(path/'valid_y.pkl').save(valid_y)


# In[21]:


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[22]:


fi = rf_feat_importance(m, xs)
plot_fi(fi[:30])


# In[26]:


len(xs.KernelName.unique())


# In[8]:


xs.shape


# In[9]:


get_ipython().run_cell_magic('time', '', 'm = rf(xs, y)\nm_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)')


# In[11]:


m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)


# In[12]:


del m


# In[13]:


df = pd.concat([xs, valid_xs])
is_valid = np.array([0] * len(xs) + [1] * len(valid_xs))
del xs, valid_xs


# In[14]:


df.shape


# In[15]:


m = rf(df, is_valid)
rf_feat_importance(m, df)[:30]


# In[17]:


rf_feat_importance(m, df)[:30]


# In[18]:


del df


# In[ ]:


m = rf(xs, y)
print('orig', m_rmse(m, valid_xs, valid_y))

for c in ('TotalFlops','SizeJ','SizeL'):
    m = rf(xs_filt.drop(c,axis=1), y)
    print(c, m_rmse(m, valid_xs.drop(c,axis=1), valid_y))


# In[ ]:





# In[ ]:





# In[15]:


m = rf(xs, y, min_samples_leaf=40);
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# In[16]:


m = rf(xs, y, min_samples_leaf=50);
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# In[18]:


m = rf(xs, y, min_samples_leaf=30);
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# In[50]:


xs.TotalFlops.hist()


# In[37]:


xs.TotalFlops.describe()


# In[55]:


filt = xs['TotalFlops'] < (0.2 * 1e12)
xs_filt = xs[filt]
y_filt = y[filt]


# In[58]:


m = rf(xs, y, min_samples_leaf=30);
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# In[60]:





# In[68]:


xs.AssertFree0ElementMultiple.value_counts(sort=False).plot.barh()


# In[11]:


from sklearn.inspection import plot_partial_dependence


# In[ ]:





# In[73]:


xs['dep_var'] = np.expm1(y)


# In[76]:


xs.drop('dep_var', axis=1, inplace=True)


# In[74]:


fig,ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(m, xs, ['AssertFree0ElementMultiple', 'dep_var'],
                        grid_resolution=20, ax=ax);


# In[6]:


df = pd.concat([xs, valid_xs])
is_valid = np.array([0] * len(xs) + [1] * len(valid_xs))


# In[7]:


del xs, valid_xs


# In[12]:


m = rf(df, is_valid, max_samples=3_000_000)
rf_feat_importance(m, df)[:30]


# In[13]:


del df


# In[19]:


xs = (path/'train_xs.pkl').load()
y = (path/'train_y.pkl').load()
valid_xs = (path/'valid_xs.pkl').load()
valid_y = (path/'valid_y.pkl').load()


# In[22]:


m = rf(xs, y)
print('orig', m_rmse(m, valid_xs, valid_y))

for c in ('TotalFlops','SizeJ','SizeL', 'SolutionName'):
    del m
    m = rf(xs.drop(c,axis=1), y)
    print(c, m_rmse(m, valid_xs.drop(c,axis=1), valid_y))


# In[23]:


del m
drop_cols = ['TotalFlops', 'SizeJ', 'SizeL', 'SolutionName']
xs = xs.drop(drop_cols, axis=1)
valid_xs = valid_xs.drop(drop_cols, axis=1)


# In[24]:


m = rf(xs, y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)


# In[25]:


rf_feat_importance(m, xs)[:30]


# In[26]:


df = pd.concat([xs, valid_xs])
is_valid = np.array([0] * len(xs) + [1] * len(valid_xs))
del m
m = rf(df, is_valid)
rf_feat_importance(m, df)[:30]


# In[ ]:





# In[ ]:





# In[14]:


to = (path/'rf_to.pkl').load()
xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y
y, valid_y = np.log1p(y), np.log1p(valid_y)

filt = xs['TotalFlops'] < (0.2 * 1e12)
xs_filt = xs[filt]
y_filt = y[filt]


# In[16]:


xs_filt = xs_filt.reset_index(drop=True)
xs_filt.head()


# In[17]:


m = rf(xs_filt, y_filt)
print('orig', m_rmse(m, valid_xs, valid_y))

for c in ('TotalFlops','SizeJ','SizeL'):
    m = rf(xs_filt.drop(c,axis=1), y)
    print(c, m_rmse(m, valid_xs.drop(c,axis=1), valid_y))


# In[18]:


print('orig', m_rmse(m, valid_xs, valid_y))


# In[ ]:


# (path/'train_xs.pkl').save(trn_xs)
# (path/'train_y.pkl').save(trn_y)
# (path/'valid_xs.pkl').save(valid_xs)
# (path/'valid_y.pkl').save(valid_y)


# In[ ]:


idxs = np.random.permutation(len(xs))
num_subgp = int(len(xs) * 0.3)
trn_idxs = idxs[:num_subgp]
len(trn_idxs)

trn_xs = xs.loc[trn_idxs].copy()
trn_y = y[trn_idxs].copy()

trn_xs.reset_index(drop=True, inplace=True)


# In[ ]:


cols = ['ThreadTile', 'WorkGroup', 'MIWaveGroup', 'MIWaveTile',
'GuaranteeNoPartialA', 'MatrixInstruction', 'SolutionName',
'GuaranteeNoPartialB', 'PrefetchLocalRead', 'LVPA',
'NumLoadsPerpendicularB', 'SizeI', '1LDSBuffer', 'LVCB', 'NumLoadsA',
'SubGroupA', 'LSPB', 'SizeL', 'GlobalLoadVectorWidthB', 'LDC',
'LoopUnroll', 'LDB', 'WorkGroupMapping', 'LdsBlockSizePerPadA',
'GlobalLoadVectorWidthA', 'LVCA', '_UseSgprForGRO',
'NumLoadsPerpendicularA', 'LdsOffsetA_Blk', 'LVPB',
'StoreRemapVectorWidth', 'LoopIters', 'LdsPadA', 'LdsBlockSizePerPad',
'PT_IndexUnrollB', 'LdsOffsetB', 'NumLoadsCoalescedB', 'SizeK',
'MatrixInstM', 'ThreadTileB', 'LSCA', 'TotalFlops', 'LSPA',
'LdsNumElementsAlignedB', 'MatrixInstB', 'NumLoadsCoalescedA',
'LdsPadB', 'MatrixInstK', 'LSCB', 'MacroTileA', 'PT_IndexUnrollA',
'AssertFree0ElementMultiple', 'StaggerUStride', 'GlobalReadVectorWidth',
'SubGroupB', 'LdsNumElements', 'LdsOffsetB_Blk', 'ThreadTileA',
'_staggerStrideShift', 'NumElementsPerThread', 'MacroTileB', 'LDA',
'NumLoadsB', 'SizeJ', 'StaggerU']
xs, valid_xs = xs[cols], valid_xs[cols]


# In[ ]:




