import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

def read_metadata(meta_path):
    metadata = pd.read_csv(meta_path)
    metadata.set_index('participant_id', inplace=True)
    return metadata


def format(file, metadata):
    dir_path = "../data/"

    mri = pd.read_csv(dir_path + "train_tsv/" + file, sep='\t', header=None)
    pattern = r'(?<=-)[^_]+'
    ID = re.search(pattern, file).group()
    data = metadata.loc[ID]

    upper_indices = np.triu_indices(mri.shape[0], k=1)
    upper_data = mri.values[upper_indices]

    mri_data = pd.Series(upper_data)
    data = pd.concat([data, mri_data])

    return data


def show_statistics(metadata: pd.DataFrame):
    missing = metadata.isna().mean() * 100
    no_people_valid = metadata.notna().all(axis=1).mean() * 100
    no_people_1NA = (metadata.isna().sum(axis=1) <= 1).mean() * 100
    print(missing)
    print()
    print(f"People with all data: {no_people_valid:.2f}%")
    print(f"People with maximum one data missing: {no_people_1NA:.2f}%")

    for column in metadata.columns:
        metadata[column] = metadata[column].fillna(metadata[column].mode()[0])
        plt.figure(figsize=(8, 6))
        sns.histplot(metadata[column], bins=10, kde=False)
        plt.title(f'Histograma pentru coloana {column}')
        plt.show()




