import pandas as pd
from pathlib import Path
import time


def get_cols(df):
    cols = df.columns
    drop_cols = [o for o in cols if len(df[o].unique()) <= 1]
    return list(set(cols) - set(drop_cols))


if __name__ == '__main__':
    start = time.time()
    path = Path('data')

    dirs = ['rocblas_hpa_hgemm_nn_inc1_asm_full', 'rocblas_hpa_hgemm_nt_inc1_asm_full', 'rocblas_hpa_hgemm_tn_inc1_asm_full']

    train_files, test_files = [], []
    for o in dirs:
        train_files += list((path/o).glob("**/*_train_raw.csv"))
        test_files += list((path/o).glob("**/*_test_raw.csv"))

    dfs = [pd.read_csv(f, low_memory=False) for f in train_files]
    train_df = pd.concat(dfs, ignore_index=True)
    cols = get_cols(train_df)
    train_df[cols].to_csv(path/'train_raw.csv', index=False)

    del dfs, train_df
    
    dfs = [pd.read_csv(f, low_memory=False) for f in test_files]
    test_df = pd.concat(dfs, ignore_index=True)
    test_df[cols].to_csv(path/'test_raw.csv', index=False)

    end = time.time()
    print("Prepare data done in {} seconds.".format(end - start))

