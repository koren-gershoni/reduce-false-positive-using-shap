import torch
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def handle_df_features(config, df):
    columns = list(df.columns.copy())
    columns.remove('label')
    if config.random_num_of_features_enabled:
        random.shuffle(columns)
        n = np.random.randint(config.min_num_of_features, config.max_num_of_features + 1)
        columns = columns[:n] + ['label']
        return df.copy()[columns]
    return df.copy()

def handle_df_examples(config, df):
    return df.copy().sample(frac=config.model_frac, replace=config.sample_with_replacement).reset_index(drop=True)

def handle_categorical_columns(df):
    if isinstance(df.columns[0], int):
        return df
    cat_cols = [col for col in df.columns if "(s)" in col]
    df_copy = df.copy()
    for col in cat_cols:
        dummies = pd.get_dummies(df_copy[col])
        dummies.columns = [f'{col}_{name}' for name in dummies.columns]
        dummies /= len(np.unique(df_copy[col]))
        df_copy.drop(col, axis=1, inplace=True)
        df_copy = pd.concat([df_copy, dummies], axis=1)
        # col_count = df[col].value_counts()
        # value_mapper = {name: (index + 1)/len(col_count) for index, name in enumerate(col_count.index.tolist())}
        # df_copy[col] = df_copy[col].map(value_mapper)
    return df_copy

def split_train_test_kdd(X_df, y_df, label_map):
    X_train = X_df[y_df == label_map["normal"]]
    y_train = y_df[y_df == label_map["normal"]]
    X_test = X_df[y_df != label_map["normal"]]
    y_test = y_df[y_df != label_map["normal"]]

    return X_train, y_train, X_test, y_test

def split_train_test_credit(X_df, y_df):
    X_train = X_df[y_df == 0]
    y_train = y_df[y_df == 0]
    X_test = X_df[y_df == 1]
    y_test = y_df[y_df == 1]

    return X_train, y_train, X_test, y_test

def split_train_val(X_df, y_df, seed=None):
    X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, stratify=y_df, random_state=seed)
    return X_train, X_val, y_train, y_val


def split_x_y(df):
    X_df = df.drop('label', axis=1)
    y_df = df['label']
    return X_df, y_df


def load_data(dataset, seed):
    label_map = None
    if dataset == "kdd":
        lines = open('datasets/kdd/kddcup.names').read().split('\n')
        label_map = dict(zip(lines[0].replace(".", "").split(","), range(len(lines[0].split(",")))))
        columns = []
        for line in lines[1:]:
            if not line:
                continue
            name = line[:line.index(":")]
            if "continuous" in line:
                name += " (c)"
            else:
                name += " (s)"
            columns.append(name)
        columns.append('label')
        df = pd.read_csv('datasets/kdd/kddcup.data_10_percent.gz', compression='gzip', sep=",", header=None,
                         names=columns)
        print(df['label'].str.replace(".", "").value_counts())
        df['label'] = df['label'].str.replace(".", "").map(label_map)
        orig_df = handle_categorical_columns(df)
        X_df, y_df = split_x_y(orig_df)
        X_train, y_train, X_test, y_test = split_train_test_kdd(X_df, y_df, label_map)
        X_train, X_val, y_train, y_val = split_train_val(X_train, y_train, seed)
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_test[X_test > 1] = 1
        X_val = scaler.transform(X_val)
        X_val[X_val > 1] = 1
    elif dataset == "creditcard":
        df = pd.read_csv('datasets/creditcard/creditcard.csv').drop('Time', axis=1)
        df = df.rename(columns={'Class': 'label'})
        orig_df = handle_categorical_columns(df)
        X_df, y_df = split_x_y(orig_df)
        X_train, y_train, X_test, y_test = split_train_test_credit(X_df, y_df)
        X_train, X_val, y_train, y_val = split_train_val(X_train, y_train, seed)
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_test[X_test > 1] = 1
        X_val = scaler.transform(X_val)
        X_val[X_val > 1] = 1
    else:
        matrix = loadmat(f'./datasets/{dataset}/{dataset}.mat')
        X_df = pd.DataFrame(matrix['X'])
        y_df = matrix['y'].reshape(-1)
        df = X_df.copy()
        df['label'] = y_df
        X_train, y_train, X_test, y_test = split_train_test_credit(X_df, y_df)
        X_train, X_val, y_train, y_val = split_train_val(X_train, y_train, seed)
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_test[X_test > 1] = 1
        X_val = scaler.transform(X_val)
        X_val[X_val > 1] = 1

    return df, X_train, y_train, X_val, y_val, X_test, y_test, label_map