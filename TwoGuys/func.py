import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

def read_metadata(meta_path):
    metadata = pd.read_csv(meta_path)
    print("(System): file loaded")
    metadata.set_index('participant_id', inplace=True)
    return metadata


def write_metadata(meta_path, metadata: pd.DataFrame):
    metadata.to_csv(meta_path)
    print("(System): file saved")


def fill_metadata(metadata: pd.DataFrame):
    prob = metadata[['ethnicity', 'race']].value_counts(normalize=True)
    columns_data = metadata[['ethnicity', 'race']].copy()
    temp = columns_data.isna().values
    col1e = temp[:, 0]
    col2e = temp[:, 1]
    val_completat = np.random.choice(prob.index, p=prob.values, size=(col1e & col2e).sum())
    val_completat = list(list(x) for x in val_completat)
    columns_data[col1e & col2e] = val_completat
    #metadata[['ethnicity', 'race']] = columns_data

    col1 = 'ethnicity'
    col2 = 'race'
    for i in range(2):
        for valoare_col in columns_data[col1].unique():
            if pd.isna(valoare_col):
                continue
            subset = columns_data[columns_data[col1] == valoare_col]

            distributie = subset[col2].value_counts(normalize=True)

            while sum(distributie) < 1.0:
                distributie /= distributie.sum()

            valori_umplere = np.random.choice(distributie.index, p=distributie.values, size=subset[col2].isna().sum())

            columns_data.loc[(columns_data[col1] == valoare_col) & (columns_data[col2].isna()), col2] = valori_umplere
        col1 = 'race'
        col2 = 'ethnicity'

    metadata[[col2, col1]] = columns_data

    for column in metadata.columns:
        if metadata[column].isna().values.any():
            prob = metadata[column].value_counts(normalize=True)
            column_data = metadata[column].copy()
            val_completat = np.random.choice(prob.index, p=prob.values, size=column_data.isna().sum())
            column_data[column_data.isna()] = val_completat
            metadata[column] = column_data

    return metadata


def format(file, metadata):
    dir_path = "../data/"

    mri = pd.read_csv(dir_path + "train_tsv/" + file, sep='\t', header=None)
    pattern = r'(?<=-)[^_]+'
    ID = re.search(pattern, file).group()
    data = metadata.loc[ID]
    if 'age' in data.index:
        y = data.pop('age')
    else:
        y = None

    upper_indices = np.triu_indices(mri.shape[0], k=1)
    upper_data = mri.values[upper_indices]

    mri_data = pd.Series(upper_data)
    data = pd.concat([data, mri_data])

    return [data.values.tolist()], [float(y)], ID


def show_statistics(metadata: pd.DataFrame, hist=False):
    missing = metadata.isna().mean() * 100
    no_people_valid = metadata.notna().all(axis=1).mean() * 100
    no_people_1NA = (metadata.isna().sum(axis=1) <= 1).mean() * 100
    print(missing)
    print()
    print(f"People with all data: {no_people_valid:.2f}%")
    print(f"People with maximum one data missing: {no_people_1NA:.2f}%")

    if hist:
        for column in metadata.columns:
            metadata[column] = metadata[column].fillna(metadata[column].mode()[0])
            plt.figure(figsize=(8, 6))
            sns.histplot(metadata[column], bins=10, kde=False)
            plt.title(f'Histograma pentru coloana {column}')
            plt.show()


def mapping_data(metadata: pd.DataFrame):
    mappings = {}
    for col in metadata.select_dtypes(include=['object', 'string']).columns:
        metadata[col], categories = pd.factorize(metadata[col])
        #print(dict(zip(range(len(categories)), categories)))
        mappings[col] = dict(zip(range(len(categories)), categories))

    return mappings


