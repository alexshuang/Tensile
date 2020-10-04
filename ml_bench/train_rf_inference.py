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
import sys

pd.options.display.max_rows = None
pd.options.display.max_columns = None


def init_df(df):
    #df.fillna(0, inplace=True)
    #import pdb; pdb.set_trace()
    df['PadA'] = df.apply(lambda x: x.LDA - x.SizeL if x.PT_TransposeA else x.LDA - x.SizeI, axis=1)
    df['PadB'] = df.apply(lambda x: x.LDB - x.SizeJ if x.PT_TransposeB else x.LDB - x.SizeL, axis=1)
    df['PadC'] = df['LDC'] - df['SizeI']
    df['PadD'] = df['LDD'] - df['SizeI']


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


def final_rf(xs, y, n_estimators=160,
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
    src = Path(sys.argv[1])
    solution_names = Path(sys.argv[2]).load()

    dep_var = 'GFlops'

    print(f"read source: {src} ...")
    test_df = pd.read_csv(src, low_memory=False)
    print("shape: {}".format(test_df.shape))

    print("############################################")
    for k, v in test_df.items():
        print("{}: {}".format(k, v.unique()))
    print("############################################")
    print("")

    init_df(test_df)
    test_df['_UseSgprForGRO'] = test_df['_UseSgprForGRO'].replace('False', 0).replace('1', 1).replace('0', 0).astype('int64')
    sizes = {
        'ThreadTile': ['1_16', '1_32', '2_16', '1_64', '2_32', '2_64', '1_80', '1_96', '1_160', '2_80', '5_32', '2_128', '4_64', '1_288', '2_144', '4_128', '4_144'],
        'WorkGroup': ['32_8_1', '64_4_1', '128_2_1' ],
        'MIWaveGroup': ['1_4', '2_2', '4_1'],
        'MIWaveTile': ['1_1', '1_2', '2_1', '1_3', '2_2', '1_5', '5_1', '2_4', '4_2', '1_9', '2_5', '4_4', '2_9', '4_9'],
        'MIBlock': ['16_16_16_1_1_1', '32_32_4_2_2_1', '32_32_8_1_1_1'],
    }
    for k, v in sizes.items():
        set_cat_order(test_df, k, v)

    procs = [Categorify, FillMissing]
    cont,cat = cont_cat_split(test_df, 1, dep_var=dep_var)
    to = TabularPandas(test_df, procs, cat, cont, y_names=dep_var)
    print("to.train: {}, to.valid: {}".format(len(to.train), len(to.valid)))

    print(to.show(3))

    xs, y = to.train.xs, to.train.y
    y = np.log1p(y)

    cols = pickle.load(open(path/'final_cols.pkl', 'rb'))
    xs = xs[cols].copy()

#    (path/'test_xs_final.pkl').save(xs)
#    (path/'test_y_final.pkl').save(y)
#    else:
#        xs = (path/'test_xs_final.pkl').load()
#        y = (path/'test_y_final.pkl').load()

    print("loading checkpoint ...")
    m = (path/'models/rf.pkl').load()

    print("test rmse: {}".format(m_rmse(m, xs, y)))
    preds = m.predict(xs) 

    #import pdb; pdb.set_trace()

    num_solutions = len(solution_names)
    target = np.array(y).reshape(-1, num_solutions)
    preds = preds.reshape(-1, num_solutions)

    num_keep = int(num_solutions * 0.2)
    top1, top2 = target.argsort()[:, -1], target.argsort()[:, -2]
    keep = preds.argsort()[:, -num_keep:]

    for i, (p, t1, t2) in enumerate(zip(keep, top1, top2)):
        if not t1 in p:
            print(f"problem size #{i} top1 not hit")
        if not t2 in p:
            print(f"problem size #{i} top2 not hit")

#    for p, t1, t2 in zip(keep, top1, top2):
#        if not t1 in p or not t2 in p:

#    import pdb; pdb.set_trace()

