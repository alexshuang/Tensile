#!/usr/bin/env python
# coding: utf-8


#from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import *
#from IPython.display import Image, display_svg, SVG
from sklearn.tree import export_graphviz
from scipy.cluster import hierarchy as hc

pd.options.display.max_rows = None
pd.options.display.max_columns = None


def single_tree(xs, y, xs2, y2, min_samples_leaf=5):
    m = DecisionTreeRegressor()
    m.fit(xs, y);
    print(r_mse(m.predict(xs), y), r_mse(m.predict(xs2), y2))


def rf_feat_imp(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,8), fontsize=6, legend=False)


def benmark_parameters(xs, y):
    print("benchmark min_samples_leaf ...")
    m = rf(xs, y, min_samples_leaf=5)
    print("min_samples_leaf=5: train rmse: {}, valid rmse: {}, oob rmse: {}".format(m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)))
    m = rf(xs, y, min_samples_leaf=10)
    print("min_samples_leaf=10: train rmse: {}, valid rmse: {}, oob rmse: {}".format(m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)))

    print("benchmark max_features ...")
    m = rf(xs, y, min_samples_leaf=5)
    print("max_features=0.3: train rmse: {}, valid rmse: {}, oob rmse: {}".format(m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)))


def cluster_columns(df, savefig=None):
    cor = np.round(scipy.stats.spearmanr(df).correlation, 6)
    cor_condensed = hc.distance.squareform(1 - cor)
    z = hc.linkage(cor_condensed, method='average')
    fig = plt.figure(figsize=(14, 14))
    dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=6)
    plt.show()
    if savefig:
        plt.savefig(savefig)


def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))


def set_cat_order(df, col, val):
    df[col] = df[col].astype('category')
    df[col].cat.set_categories(val, ordered=True, inplace=True)
    print(col, df[col].unique())


def reorder_str_cols(df):
    print("############################################")
    cols = list(set([n for n,c in df.items() if is_string_dtype(c)]) - set(['SolutionName']))
    for c in cols:
        s = list(df[c].unique())
        ss = np.array([reduce(lambda x, y: x * y, [int(n) for n in o.split('_')]) for o in s])
        s_sorted = [s[i] for i in ss.argsort()]
        set_cat_order(df, c, s_sorted)
    print("############################################")
    print("")


def rf(xs, y, n_estimators=40, max_samples=1_000_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


def final_rf(xs, y, n_estimators=40,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


def get_oob(x, y):
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5,
        max_samples=1_000_000, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(x, y)
    return m.oob_score_


if __name__ == '__main__':
    path = Path('data')
    img_path = path/'imgs'
    img_path.mkdir(exist_ok=True)

    print(list(path.glob('*.csv')))

    # ## Look at Data

    dep_var = 'GFlops'

    if not (path/'rf_to.pkl').is_file():
        train_df = pd.read_csv(path/'train_raw.csv', low_memory=False)
        valid_df = pd.read_csv(path/'test_raw.csv', low_memory=False)
        train_df = train_df[~(train_df.GFlops <= 0)].reset_index(drop=True)
        train_df.fillna(0, inplace=True)
        valid_df = valid_df[~(valid_df.GFlops <= 0)].reset_index(drop=True)
        valid_df.fillna(0, inplace=True)
        df = pd.concat([train_df, valid_df], ignore_index=True).reset_index(drop=True)
        df['_UseSgprForGRO'] = df['_UseSgprForGRO'].replace('False', 0).replace('1', 1).replace('0', 0).astype('int64')

        sizes = {
            'ThreadTile': ['1_16', '1_32', '2_16', '1_64', '2_32', '2_64', '1_80', '1_96', '1_160', '2_80', '5_32', '2_128', '4_64', '1_288', '2_144', '4_128', '4_144'],
            'WorkGroup': ['32_8_1', '64_4_1', '128_2_1' ],
            'MIWaveGroup': ['1_4', '2_2', '4_1'],
            'MIWaveTile': ['1_1', '1_2', '2_1', '1_3', '2_2', '1_5', '5_1', '2_4', '4_2', '1_9', '2_5', '4_4', '2_9', '4_9'],
            'MIBlock': ['16_16_16_1_1_1', '32_32_4_2_2_1', '32_32_8_1_1_1'],
        }
        for k, v in sizes.items():
            set_cat_order(df, k, v)

        procs = [Categorify]
        cont,cat = cont_cat_split(df, 1, dep_var=dep_var)
        splits = (list(np.arange(0, len(train_df))), list(np.arange(len(train_df), len(df))))
        to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
        print("to.train: {}, to.valid: {}".format(len(to.train), len(to.valid)))
        (path/'rf_to.pkl').save(to)
    else:
        to = (path/'rf_to.pkl').load()

    xs, y = to.train.xs.copy(), to.train.y.copy()
    valid_xs, valid_y = to.valid.xs.copy(), to.valid.y.copy()
    y, valid_y = np.log1p(y), np.log1p(valid_y)
#    train_keep = ~(y < 0)
#    y = y[train_keep]
#    xs = xs[train_keep].copy().reset_index(drop=True)
#    valid_keep = ~(valid_y < 0)
#    valid_y = valid_y[valid_keep]
#    valid_xs = valid_xs[valid_keep].copy().reset_index(drop=True)
#    xs.fillna(0, inplace=True)
#    valid_xs.fillna(0, inplace=True)

    #import pdb; pdb.set_trace()

    #single_tree(xs, y, valid_xs, valid_y, min_samples_leaf=25)
    m = final_rf(xs, y, min_samples_leaf=25)
    print("min_samples_leaf=25: train rmse: {}, valid rmse: {}, oob rmse: {}".format(m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)))

#    cluster_columns(xs, f'data/imgs/cluster_cols.png')

    final_cols = ['ThreadTile', 'WorkGroup', 'MIWaveGroup', 'MIWaveTile',
                'GuaranteeNoPartialA', 'MatrixInstruction',
                'GuaranteeNoPartialB', 'PrefetchLocalRead', 'LVPA',
                'NumLoadsPerpendicularB', 'SizeI', '1LDSBuffer', 'LVCB', 'NumLoadsA',
                'SubGroup0', 'LSPB', 'SizeL', 'GlobalLoadVectorWidthB', 'LDC',
                'LoopUnroll', 'LDB', 'WorkGroupMapping', 'LdsBlockSizePerPadA', 
                'GlobalLoadVectorWidthA', 'LVCA', '_UseSgprForGRO',
                'NumLoadsPerpendicularA', 'LdsOffsetA_Blk', 'LVPB',
                'StoreRemapVectorWidth', 'LoopIters', 'PT_TransposeA', 'LdsBlockSizePerPad',
                'PT_TransposeB', 'LdsOffsetB', 'NumLoadsCoalescedB', 'SizeK',
                'MatrixInstM', 'ThreadTileB', 'LSCA', 'TotalFlops', 'LSPA',
                'LdsNumElementsAlignedB', 'MatrixInstB', 'NumLoadsCoalescedA',
                'LdsPadB', 'MatrixInstK', 'LSCB', 'MacroTileA', 'PT_IndexUnrollA',
                'AssertFree0ElementMultiple', 'StaggerUStride', 'GlobalReadVectorWidth', 'SubGroup0', 'LdsNumElements', 'LdsOffsetB_Blk', 'ThreadTileA',
                '_staggerStrideShift', 'NumElementsPerThread', 'MacroTileB', 'LDA',
                'NumLoadsB', 'SizeJ', 'StaggerU']

    xs, valid_xs = xs[final_cols].copy(), valid_xs[final_cols].copy()
    m = final_rf(xs, y, min_samples_leaf=25)
    print("drop cols: train rmse: {}, valid rmse: {}, oob rmse: {}".format(m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)))

    fi = rf_feat_imp(m, xs)
    plot_fi(fi)
    plt.savefig(img_path/'feat_imp.png')

    cluster_columns(xs, f'data/imgs/cluster_cols2.png')

    cols = list(fi[fi.imp < 0.001].cols)
    oobs = {c:get_oob(xs.drop(c, axis=1), y) for c in cols}
    (path/'oobs.pkl').save(oobs)

    import pdb; pdb.set_trace()

    m = final_rf(xs, y, min_samples_leaf=25)
    print("min_samples_leaf=25: train rmse: {}, valid rmse: {}, oob rmse: {}".format(m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)))

    #benmark_parameters(xs, y)

    if not (path/'models/rf.pkl').is_file():
        print("start train with final_rf ...")
        m = final_rf(xs, y)
        print("train rmse: {}, valid rmse: {}, oob rmse: {}".format(m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)))
    else:
        print("loading checkpoint of model ...")
        m = (path/'models/rf.pkl').load()

    print("train rmse: {}, valid rmse: {}, oob rmse: {}".format(m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y), r_mse(m.oob_prediction_, y)))
    preds = m.predict(valid_xs), 
    print("prediction errors: mean: {}, median: {}".format(np.abs(np.expm1(preds[0]) - np.expm1(valid_y)).mean(), np.abs(np.expm1(preds[0]) - np.expm1(valid_y)).median()))
    print("real: {}".format(list(np.expm1(valid_y.values))[:10]))
    print("prediction: {}".format(list(np.expm1(preds[0]))[:10]))

    exit(0)

    fi = rf_feat_imp(m, xs)
    plot_fi(fi)
    plt.savefig(img_path/'feat_imp.png')

    import pdb; pdb.set_trace()

