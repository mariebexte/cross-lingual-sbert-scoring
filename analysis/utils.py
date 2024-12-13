import pandas as pd


def read_data(path, answer_column, target_column):

    df = pd.read_csv(path)
    df = df.fillna('')
    df[answer_column] = df[answer_column].astype(str)
    df[target_column] = df[target_column].astype(int)

    return df