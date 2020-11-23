#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_bool_dtype, is_numeric_dtype, is_object_dtype, is_categorical_dtype, is_integer_dtype, is_float_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from IPython.display import Image, display_svg, SVG
from dtreeviz.trees import *
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


# In[2]:


torch.cuda.is_available()


# ## Look at Data

# In[3]:


path = Path('data/small/')
img_path = path/'imgs'
model_path = path/'models'
img_path.mkdir(exist_ok=True)
model_path.mkdir(exist_ok=True)


# In[6]:


train_df = pd.read_feather(path/'train.feat')
valid_df = pd.read_feather(path/'valid.feat')


# In[7]:


print(f"train_df: {train_df.shape}, valid_df: {valid_df.shape}")


# In[8]:


((train_df.GFlops <= 0).sum() / len(train_df) * 100), ((valid_df.GFlops <= 0).sum() / len(valid_df) * 100)


# In[9]:


train_df = train_df[train_df['GFlops'] > 0].reset_index(drop=True)
valid_df = valid_df[valid_df['GFlops'] > 0].reset_index(drop=True)
len(train_df), len(valid_df)


# In[10]:


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
    nn_idxs = np.where(df['ProblemType'].apply(lambda x: 'Ailk_Bljk' in x))
    tn_idxs = np.where(df['ProblemType'].apply(lambda x: 'Alik_Bljk' in x))
    nt_idxs = np.where(df['ProblemType'].apply(lambda x: 'Ailk_Bjlk' in x))
    df.loc[nn_idxs]['PadA'] = df['LDA'] - df['SizeI']
    df.loc[nn_idxs]['PadB'] = df['LDB'] - df['SizeL']
    df.loc[nn_idxs]['PadC'] = df['LDC'] - df['SizeI']
    df.loc[tn_idxs]['PadA'] = df['LDA'] - df['SizeL']
    df.loc[nt_idxs]['PadB'] = df['LDB'] - df['SizeJ']
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


# In[11]:


get_ipython().run_cell_magic('time', '', 'preproc_df(train_df)\npreproc_df(valid_df)\ntrain_cats(train_df)\napply_cats(valid_df, train_df)\ncategorify(train_df)\ncategorify(valid_df)')


# ## Ranking

# In[12]:


train_df.drop(['GFlops'], axis=1, inplace=True)
valid_df.drop(['GFlops'], axis=1, inplace=True)
dep_var = 'Ranking'
y, valid_y = np.log1p(train_df[dep_var].values), np.log1p(valid_df[dep_var].values)
xs = train_df.drop(dep_var, axis=1)
valid_xs = valid_df.drop(dep_var, axis=1)


# In[13]:


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
    
def rf(xs, y, n_estimators=40, max_features=0.5, min_samples_leaf=25, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  max_samples=200_000, oob_score=True, **kwargs).fit(xs, y)

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def plot_fi(fi, fig=img_path/'fi.png'):
    fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
    plt.savefig(fig)


# In[14]:


get_ipython().run_cell_magic('time', '', 'm = rf(xs, y)\neval_model(m, xs, y, valid_xs, valid_y)')


# In[16]:


fi = rf_feat_importance(m, xs)
plot_fi(fi[:30])


# In[17]:


get_ipython().run_cell_magic('time', '', 'to_keep = fi[fi.imp > 0.001].cols\nxs_keep, valid_xs_keep = xs[to_keep], valid_xs[to_keep]\ndel m\nm = rf(xs_keep, y)\neval_model(m, xs_keep, y, valid_xs_keep, valid_y)')


# In[18]:


len(xs_keep.columns), len(xs.columns), xs_keep.columns


# In[19]:


del m
cluster_columns(xs_keep, figsize=(12, 14), font_size=9);


# In[37]:


dup_cols = ['MIBlock_0', 'MIBlock_1', 'MatrixInstruction_1',
             'NumGlobalWriteVectorsPerThread', 'NumElementsPerThread',
             'MIWaveTile_0', 'ThreadTile_0',
             'MIWaveTile_1', 'ThreadTile_1',
             'LdsOffsetB', 'LdsNumElementsAlignedA',
             'DepthU', 'LoopUnroll',
             'MatrixInstruction_2', 'MatrixInstK',
             'SizeI', 'LDC']
#              'LdsBlockSizePerPadB',
#              'PT_IndexAssignmentsB_0',
#              'MatrixInstruction_2', 'MIBlock_2']
#              'PT_IndexAssignmentsB_1', 'PT_IndexUnrollB', 'PT_TLUB', 

def get_oob(x, y):
    m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.5,
                                  min_samples_leaf=25,
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


# In[20]:


get_ipython().run_cell_magic('time', '', "# drop_cols = ['MIBlock_0', 'MatrixInstruction_1',\n#              'NumGlobalWriteVectorsPerThread',\n#              'MIWaveTile_0',\n#              'MIWaveTile_1',\n#              'PT_IndexAssignmentsB_1', 'PT_IndexUnrollB', 'PT_TLUB', \n#              'LdsNumElementsAlignedA',\n#              'DepthU',\n#              'LdsBlockSizePerPadB',\n#              'PT_IndexAssignmentsB_0',\n#              'MatrixInstruction_2', 'MIBlock_2']\ndrop_cols = ['MIBlock_0', 'MIBlock_1',\n             'NumElementsPerThread',\n             'MIWaveTile_0',\n             'MIWaveTile_1',\n             'LdsOffsetB',\n             'LoopUnroll',\n             'MatrixInstK',\n             'SizeI'\n             ]\nxs_keep2, valid_xs_keep2 = xs_keep.copy(), valid_xs_keep.copy()\nfor c in drop_cols:\n    if c in xs_keep2 and c in valid_xs_keep2:\n        xs_keep2.drop(c, axis=1, inplace=True)\n        valid_xs_keep2.drop(c, axis=1, inplace=True)\n\nm = rf(xs_keep2, y)\neval_model(m, xs_keep2, y, valid_xs_keep2, valid_y)")


# In[39]:


fi = rf_feat_importance(m, xs_keep2)
plot_fi(fi[:30])


# In[28]:


get_ipython().run_cell_magic('time', '', 'to_keep = fi[fi.imp > 0.001].cols\nxs_keep3, valid_xs_keep3 = xs_keep2[to_keep], valid_xs_keep2[to_keep]\ndel m\nm = rf(xs_keep3, y)\neval_model(m, xs_keep3, y, valid_xs_keep3, valid_y)')


# In[31]:


del xs_keep, valid_xs_keep, xs_keep3, valid_xs_keep3


# In[40]:


xs_final, valid_xs_final = xs_keep2.copy(), valid_xs_keep2.copy()
del xs_keep2, valid_xs_keep2


# In[42]:


len(xs_final.columns), xs_final.columns


# ## is_valid

# In[43]:


df = pd.concat([xs_final, valid_xs_final])
is_valid = np.array([0] * len(xs_final) + [1] * len(valid_xs_final))
m = rf(df, is_valid)
fi = rf_feat_importance(m, df)
fi[:20]


# In[44]:


m = rf(xs_final, y)
print('original', eval_model(m, xs_final, y, valid_xs_final, valid_y))
for c in fi[fi.imp > 0.01].cols.values:
    if c in xs_final:
        m = rf(xs_final.drop(c, axis=1), y)
        print(c, eval_model(m, xs_final.drop(c, axis=1), y, valid_xs_final.drop(c, axis=1), valid_y))


# In[45]:


# drop_cols = ['TotalFlops', 'AspectRatioB', 'AreaB', 'AoverB',
#        'AspectRatioC', 'AreaA', 'AspectRatioA', 'SizeJ',
#        'ProblemType', 'LDA', 'LdsPadB']
drop_cols = ['SolutionName']


# In[49]:


m = rf(xs_final.drop(drop_cols, axis=1), y)
print(eval_model(m, xs_final.drop(drop_cols, axis=1), y, valid_xs_final.drop(drop_cols, axis=1), valid_y))


# In[46]:


xs_final.drop(drop_cols, axis=1, inplace=True)
valid_xs_final.drop(drop_cols, axis=1, inplace=True)


# In[47]:


m = rf(xs_final, y)
eval_model(m, xs_final, y, valid_xs_final, valid_y)


# In[48]:


fi = rf_feat_importance(m, xs_final)
plot_fi(fi[:30])


# In[49]:


fi


# In[50]:


(path/'xs_final.pkl').save(xs_final)
(path/'valid_xs_final.pkl').save(valid_xs_final)
(path/'y.pkl').save(y)
(path/'valid_y.pkl').save(valid_y)


# ## Partial Dependence

# In[8]:


xs_final = (path/'xs_final.pkl').load()
valid_xs_final = (path/'valid_xs_final.pkl').load()
y = (path/'y.pkl').load()
valid_y = (path/'valid_y.pkl').load()


# In[1]:


xs_final['AreaC'].hist(bins=50);


# In[70]:


xs_final['NumElementsPerThread'].value_counts().plot(kind='barh');


# In[71]:


xs_final['SizeL'].hist(bins=50);


# In[73]:


from sklearn.inspection import plot_partial_dependence

fig,ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(m, valid_xs_final, ['AreaC', 'NumElementsPerThread'],
                        grid_resolution=30, ax=ax);
fig.savefig(img_path/'partial_dependence.png')


# In[82]:


((xs_final['AreaC'] > 5e6).sum() / xs_final.shape[0],
(xs_final['NumElementsPerThread'] > 150).sum() / xs_final.shape[0])


# In[95]:


keep = xs_final['AreaC'] < 5e6
xs_final2 = xs_final[keep].reset_index(drop=True)
y2 = y[np.where(keep)]

keep = valid_xs_final['AreaC'] < 5e6
valid_xs_final2 = valid_xs_final[keep].reset_index(drop=True)
valid_y2 = valid_y[np.where(keep)]


# In[96]:


keep = xs_final2['NumElementsPerThread'] < 150
xs_final3 = xs_final2[keep].reset_index(drop=True)
y3 = y2[np.where(keep)]

keep = valid_xs_final2['NumElementsPerThread'] < 150
valid_xs_final3 = valid_xs_final2[keep].reset_index(drop=True)
valid_y3 = valid_y2[np.where(keep)]


# In[99]:


m = rf(xs_final3, y3)
eval_model(m, xs_final3, y3, valid_xs_final3, valid_y3)


# In[102]:


(path/'xs_final2.pkl').save(xs_final3)
(path/'valid_xs_final2.pkl').save(valid_xs_final3)
(path/'y2.pkl').save(y3)
(path/'valid_y2.pkl').save(valid_y3)


# ## Target Mean Encoding

# In[4]:


xs_final = (path/'xs_final.pkl').load()
valid_xs_final = (path/'valid_xs_final.pkl').load()
y = (path/'y.pkl').load()
valid_y = (path/'valid_y.pkl').load()


# In[9]:


m = rf(xs_final, y)
eval_model(m, xs_final, y, valid_xs_final, valid_y)


# In[10]:


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


# In[11]:


enc_cols = ['SolutionName']
xs_final['Target'] = y
agg_op = ['mean']
dep_var = 'Target'
recipies = [(c, dep_var, agg_op) for c in enc_cols]
recipies


# In[12]:


train_new, tme = gen_target_mean_enc(xs_final, recipies)
train_new.drop(['Target'], axis=1, inplace=True)
valid_new = apply_target_mean_enc(valid_xs_final, tme)
del xs_final, valid_xs_final


# In[14]:


m = rf(train_new, y)
eval_model(m, train_new, y, valid_new, valid_y)


# ## Final

# In[5]:


xs_final = (path/'xs_final.pkl').load()
valid_xs_final = (path/'valid_xs_final.pkl').load()
y = (path/'y.pkl').load()
valid_y = (path/'valid_y.pkl').load()


# In[51]:


def final_rf(xs, y, n_estimators=500, max_features=0.5, min_samples_leaf=25, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, max_samples=200_000, **kwargs).fit(xs, y)


# In[55]:


get_ipython().run_cell_magic('time', '', 'model = final_rf(xs_final, y)\neval_model(model, xs_final, y, valid_xs_final, valid_y)')


# In[24]:


(model_path/'rf_model_final.pkl').save(model)


# ## Testing

# In[56]:


test_df = pd.read_feather(path/'test_tn_raw_full.feat')


# In[42]:


# test_df['ProblemType'] = ['Cijk_Alik_Bljk_HBH'] * len(test_df)


# In[57]:


final_cols = xs_final.columns


# In[58]:


len(final_cols), final_cols


# In[59]:


len(test_df.columns)


# In[60]:


preproc_df(test_df)
train_cats(test_df)
categorify(test_df)
test_xs = test_df[final_cols]


# In[61]:


test_xs.columns


# In[62]:


preds = model.predict(test_xs)


# In[63]:


target = pd.read_csv(path/'Cijk_Alik_Bljk_HBH_00.csv', low_memory=False)


# In[64]:


target.shape


# In[65]:


preds = preds.reshape(389, -1)


# In[109]:


num_solutions = preds.shape[1]
top = int(num_solutions * 0.1)
top_preds = preds.argsort()[:, :top]
top_preds.shape


# In[110]:


cols = target.columns[10:]
gflops = target[cols].values


# In[111]:


# peak_preds = top_preds[:, 0].reshape(-1)
# gflops_preds = [o[peak_preds[i]] for i, o in enumerate(gflops)]


# In[112]:


gflops_preds = []
for o, p in zip(gflops, top_preds):
    max_gflops = 0
    for i in p:
        if o[i] > max_gflops: max_gflops = o[i]
    gflops_preds.append(max_gflops)


# In[113]:


gflops.sort()
gflops_target = gflops[:, -1]


# In[114]:


plt.plot(np.arange(len(gflops_preds)), gflops_preds, label='preds')
plt.plot(np.arange(len(gflops_target)), gflops_target, label='target')
plt.legend();


# In[115]:


rmse(gflops_preds, gflops_target)


# In[116]:


gflops_target.mean(), gflops_target.std()


# In[117]:


plt.plot(np.arange(50), gflops_preds[50:100], label='preds')
plt.plot(np.arange(50), gflops_target[50:100], label='target')
plt.legend();


# In[118]:


plt.plot(np.arange(50), gflops_preds[100:150], label='preds')
plt.plot(np.arange(50), gflops_target[100:150], label='target')
plt.legend();


# ## NN

# In[30]:


xs_final = (path/'xs_final2.pkl').load()
valid_xs_final = (path/'valid_xs_final2.pkl').load()
y = (path/'y2.pkl').load()
valid_y = (path/'valid_y2.pkl').load()


# In[31]:


y, valid_y = np.log1p(y), np.log1p(valid_y)


# In[21]:


idxs = np.random.permutation(len(xs_final))
trn_idxs = idxs[:len(idxs)//2]
valid_idxs = np.random.permutation(len(valid_xs_final))
val_idxs = valid_idxs[:len(valid_idxs)//2]

xs_final = xs_final.loc[trn_idxs].reset_index(drop=True)
valid_xs_final = valid_xs_final.loc[val_idxs].reset_index(drop=True)
y, valid_y = y[trn_idxs], valid_y[val_idxs]


# In[23]:


df = pd.concat([xs_final, valid_xs_final], ignore_index=True)
df['Target'] = np.concatenate([y, valid_y])
splits = (list(np.arange(0, len(xs_final))), list(np.arange(len(valid_xs_final), len(df))))


# In[24]:


del xs_final, valid_xs_final, y, valid_y


# In[25]:


len(df.columns), df.columns


# In[26]:


dep_var = 'Target'
procs_nn = [Categorify, Normalize]
cat_cols = ['SolutionName']
cont_cols = list(set(df.columns) - set(cat_cols))
to_nn = TabularPandas(df, procs_nn, cat_cols, cont_cols, splits=splits, y_names=dep_var)     
# (path/f'to_nn.pkl').save(to_nn)


# In[27]:


dls = to_nn.dataloaders(512)


# In[34]:


learn = tabular_learner(dls, path=path, y_range=[0, 1], layers=[500, 250], n_out=1,
                        loss_func=F.mse_loss, metrics=rmse)


# In[35]:


learn.lr_find()


# In[36]:


learn.fit_one_cycle(7, 1e-1)


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




