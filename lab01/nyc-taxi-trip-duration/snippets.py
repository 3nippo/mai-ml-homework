from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def drop_column(column):
    global train_df, test_df

    if isinstance(column, list):
        list(map(drop_column, column))
        return

    if not isinstance(column, str):
        raise NotImplementedError

    for df in [train_df, test_df]:
        df.drop(column, axis=1, inplace=True)


def ohe_column(column):
    global train_df, test_df

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    train_df_ohed = ohe.fit_transform(train_df[[column]])
    test_df_ohed = ohe.transform(test_df[[column]])

    drop_column(column)

    result = []

    for df, ohed in zip([train_df, test_df], [train_df_ohed, test_df_ohed]):
        result.append(pd.concat(
            [pd.DataFrame(ohed), df],
            axis=1,
            ignore_index=True
        ))

        column_names = [f'{column}_{value}' for value in ohe.categories_[0]]
        column_names += df.columns.to_list()

        result[-1].columns = column_names

    return result


def describe(column=None):
    global curr_col

    column = column or curr_col

    print(f"Column: {column}")
    print(f"Train dataset uniques: {train_df[column].unique()}")
    print(f"Test dataset uniques: {test_df[column].unique()}")
    print(f"Train has NaNs: {train_df[column].isnull().sum()}")
    print(f"Test has NaNs: {test_df[column].isnull().sum()}")


target = 'Null'
curr_col = 'Null'


def plot_categoric_feature(column=None, uniques=None):
    global train_df
    global curr_col

    column = column or curr_col

    uniques = uniques or train_df[column].unique()

    fig, axs = plt.subplots(ncols=len(uniques), figsize=(20, 5))

    print(uniques)

    for value, ax in zip(uniques, axs):
        if pd.isnull(value):
            continue

        sns.countplot(
            x=target,
            hue=column,
            data=train_df[train_df[column] == value],
            ax=ax
        )
        ax.get_legend().remove()


def le_column(column=None):
    global train_df, test_df
    global curr_col

    column = column or curr_col

    le = LabelEncoder()

    train_df[column] = le.fit_transform(train_df[column])
    test_df[column] = le.transform(test_df[column])


def init(_train_df, _test_df):
    global train_df, test_df

    train_df = _train_df
    test_df = _test_df
