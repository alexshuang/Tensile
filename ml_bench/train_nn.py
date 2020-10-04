#!/usr/bin/env python
# coding: utf-8

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *

def init_df(df):
    #df.fillna(0, inplace=True)
    df['_UseSgprForGRO'] = df['_UseSgprForGRO'].replace('False', 0).replace('1', 1).astype('int64')
    df['PadA'] = df.apply(lambda x: x.LDA - x.SizeL if x.PT_TransposeA else x.LDA - x.SizeI, axis=1)
    df['PadB'] = df.apply(lambda x: x.LDB - x.SizeJ if x.PT_TransposeB else x.LDB - x.SizeL, axis=1)
    df['PadC'] = df['LDC'] - df['SizeI']
    df['PadD'] = df['LDD'] - df['SizeI']

path = Path('data/train/src')
img_path = path/'imgs'
img_path.mkdir(exist_ok=True)

print(list(path.glob('*.csv')))

# ## Look at Data

ds_chkpt_name = 'train'
model_chkpt_name = 'nn'
model_chkpt = path/f'models/{model_chkpt_name}.pth'

dep_var = 'Ranking'

if not (path/'to_nn.pkl').is_file():
    train_df = pd.read_csv(path/'train.csv', low_memory=False)
    valid_df = pd.read_csv(path/'valid.csv', low_memory=False)
    df = pd.concat([train_df, valid_df], ignore_index=True)
    df[dep_var] = np.log1p(df[dep_var])
    splits = (list(np.arange(0, len(train_df))), list(np.arange(len(train_df), len(train_df) + len(valid_df))))
    cont_cols, cat_cols = cont_cat_split(df, max_card=30000, dep_var=dep_var)
    procs_nn = [Categorify, Normalize]
    to_nn = TabularPandas(df, procs_nn, cat_cols, cont_cols,
                          splits=splits, y_names=dep_var)
    (path/f'to_nn.pkl').save(to_nn)
else:
    to_nn = (path/f'to_nn.pkl').load()

dls = to_nn.dataloaders(1024)

# train
learn = tabular_learner(dls, path=path, y_range=[0, 12], layers=[200, 100], n_out=1, loss_func=F.mse_loss, metrics=rmse)

learn.lr_find()
plt.savefig(img_path/'lr_find.png')

print("start to train ...")
learn.fit_one_cycle(2, 1e-2)
learn.save('nn_raw')

test_y = np.log1p(test_df[dep_var])
test_df.drop(dep_var, axis=1, inplace=True)

dl = learn.dls.test_dl(test_df)
preds = learn.get_preds(dl=dl)
preds = preds[0].squeeze()
print("test rmse: {}".format(rmse(preds, test_y)))

import pdb; pdb.set_trace()

preds, target = learn.get_preds()
print(rmse(preds, target))


# In[41]:

target = torch.Tensor(test_df.GFlops.values)
_test_df = test_df[cat_cols + cont_cols].copy()
test_dl = learn.dls.test_dl(_test_df)
preds = learn.get_preds(dl=test_dl)
preds = preds[0].squeeze()

def rmse(preds, target):
    return torch.sqrt(((preds - target)**2).mean()).item()

print("test rmse: {}".format(rmse(preds, target)))

import pdb; pdb.set_trace()

preds = torch.expm1(preds)
target = torch.expm1(target)
print("test rmse: {}".format(rmse(preds, target)))

