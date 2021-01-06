#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_bool_dtype, is_numeric_dtype, is_object_dtype
from sklearn.ensemble import RandomForestRegressor
import yaml
import os
import re
import math
import pickle
from collections import defaultdict
from sklearn.model_selection import KFold
from pathlib import Path
from functools import partial
import argparse


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


def eval_model(m, trn_xs, trn_y, val_xs, val_y, test_xs=None, test_y=None):
    res = [m_rmse(m, trn_xs, trn_y), m_rmse(m, val_xs, val_y)]
    if test_xs and test_y:
        res += [m_rmse(m, test_xs, test_y)]
    return res


def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))


def cluster_columns(df, figsize=(10,6), font_size=12, fig_path='cc.png'):
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


def plot_fi(fi, fig='fi.png'):
    fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
    plt.savefig(fig)


def final_rf(xs, y, n_estimators=200, max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, **kwargs).fit(xs, y)


def metric(f):
    def _inner(p, t):
        keep = t >= 0.
        p, t = p[keep], t[keep]
        return partial(f, p, t)
    return _inner


@metric
def mae(p, t): return np.mean(abs(p - t))
@metric
def mee(p, t, eff_err): return np.mean(np.abs(p - t) < (t * eff_err))
@metric
def max_ee(p, t): return np.max(np.abs(p - t) / t)
@metric
def mean_ee(p, t): return np.mean(np.abs(p - t) / t)


def testing(path, model, final_cols, n_pct=0.25, eff_err=0.015, debug=False):
    src = []
    for t in ('**/valid_N*.feat', '**/test_N*.feat'):
        src.extend(path.glob(t))
    n_pat = re.compile(r'_N(.*)')
    output = path/'train_result.csv'
    res = defaultdict(lambda: [])

    for tf in src:
        print(f"Inference {tf} ...")
        out_path = tf.parent/'train_out'
        img_path = out_path/'imgs'
        out_path.mkdir(exist_ok=True)
        img_path.mkdir(exist_ok=True)

        num_solution = re.findall(n_pat, tf.stem)[0]
        assert num_solution.isdecimal()
        num_solution = eval(num_solution)

        df = pd.read_feather(tf)
        gflops = df['GFlops'].values
        gflops = gflops.reshape(-1, num_solution)
        n = gflops.shape[0]
        topN_target = np.argsort(-gflops)[:, 0].reshape(-1)

        preproc_df(df)
        
        # rf
        rf_df = df[final_cols].copy()
        train_cats(rf_df)
        categorify(rf_df)
        rf_preds = model.predict(rf_df)
        rf_preds = np.expm1(rf_preds)
        
        preds = rf_preds.reshape(gflops.shape)
        num_preds = int(preds.shape[1] * n_pct)
        topN_preds = np.argsort(preds)[:, :num_preds]

        if debug:
            print("Dump fast_solution_indices ...")
            with (out_path/'fast_solution_indices.csv').open('w') as fp:
                for p in topN_preds:
                    fp.write(f"{','.join([str(o) for o in np.sort(p)])}\n")

        top1_acc, gflops_preds, gflops_target = [], [], []
        for i, (p, t) in enumerate(zip(topN_preds, topN_target)):
            if gflops[i, t] < 0 or t in p:
                top1_acc.append(True)
            else:
                top1_acc.append(False)

            max_gflops = 0
            for j in p:
                if gflops[i, j] > max_gflops:
                    max_gflops = gflops[i, j]
            gflops_preds.append(max_gflops)
            gflops_target.append(gflops[i, t])
        gflops_preds, gflops_target = np.array(gflops_preds), np.array(gflops_target)

        top1_acc = np.mean(top1_acc)
        eff_acc = mee(gflops_preds, gflops_target)(eff_err)
        max_eff_err = max_ee(gflops_preds, gflops_target)()
        mean_eff_err = mean_ee(gflops_preds, gflops_target)()

        print(f"{tf.stem}: {n_pct*100}%/{num_preds}/{num_solution} solutions, top1 accuracy: {top1_acc*100:.2f}%, efficiency error < 1.5%: {eff_acc*100:.2f}%, mean efficiency error: {mean_eff_err*100:.2f}%, max efficiency error: {max_eff_err*100:.2f}%")
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        x_axis = np.arange(n)
        j = n // 9
        for i, ax in enumerate(axes.flatten()):
            ax.plot(x_axis[i*j:i*j+j], gflops_preds[i*j:i*j+j], label='prediction')
            ax.plot(x_axis[i*j:i*j+j], gflops_target[i*j:i*j+j], label='target')
        plt.subplots_adjust(right=1.5)
        plt.legend()
        plt.show()
        plt.gcf().savefig(img_path/f'{tf.stem}_problem{n}_pct{n_pct}.png', dpi=600, bbox_inches='tight')

        res['Problem Count'].append(n)
        res['Kernel Count (fast-bench/origin)'].append(f'{num_preds}/{num_solution}')
        res['Top1 Accuracy Rate (%)'].append(top1_acc * 100)
        res[f'Efficiency Error < {eff_err*100}% (%)'].append(eff_acc * 100)
        res['Mean Efficiency Error (%)'].append(mean_eff_err * 100)
        res['Max Efficiency Error (%)'].append(max_eff_err * 100)

    # dump result
    df = pd.DataFrame(res)
    df.to_csv(output, float_format='%.2f', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Training with random forest"
    parser.add_argument("--path", type=str, help="Data path.")
    parser.add_argument("--train", type=str, default='train/', help="train set dir")
    parser.add_argument("--test", type=str, default='test/', help="test set dir")
    parser.add_argument("--kernel_pct", type=float, default=0.25, help="The proportion of selected topN kernels in the entire kernel set.")
    parser.add_argument("--eff_err", type=float, default=0.015, help="Accuracy error range of peak GFlops")
    parser.add_argument("--debug", action="store_true", default=False, help="Dump debug info")
    args = parser.parse_args()

    path = Path(args.path)
    train_path = path/args.train
    test_path = path/args.test
    img_path = path/'imgs'
    model_path = path/'models'
    img_path.mkdir(exist_ok=True)
    model_path.mkdir(exist_ok=True)

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

    # Training
    if not (model_path/'final_rf_model.pkl').is_file():
        print("Training model ...")
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

    print("Loading model ...")
    m = pickle.load((model_path/'final_rf_model.pkl').open('rb'))
    final_cols = pickle.load((model_path/'final_columns.pkl').open('rb'))

    print(f"Validation ...")
    testing(train_path, m, final_cols, n_pct=args.kernel_pct, eff_err=args.eff_err, debug=args.debug)

    if test_path.is_dir():
        print(f"Testing ...")
        testing(test_path, m, final_cols, n_pct=args.kernel_pct, eff_err=args.eff_err, debug=args.debug)
