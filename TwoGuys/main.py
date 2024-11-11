import pandas as pd
import os
import numpy as np

def main():
    meta_path = "../data/metadata/training_metadata.csv"
    file = "sub-NDARAA306NT2_ses-HBNsiteRU_task-rest_run-2_atlas-Schaefer2018p200n17_space-MNI152NLin6ASym_reg-36Parameter_desc-PearsonNilearn_correlations.tsv"
    metadata = pd.read_csv(meta_path)
   # format(file, metadata)
    dir_path="../data/train_tsv"
    list_files=[]
    for file in os.listdir(dir_path):
        if file.endswith(".tsv"):
            list_files.append(file)

    print(len(list_files))

def format(file, metadata):
    dir_path = "../data/"

    mri = pd.read_csv(dir_path + "train_tsv/" + file, sep='\t', header=None)
    ID = file[4:16]

    metadata.set_index('participant_id', inplace=True)
    data = metadata.loc[ID]

    upper_indices = np.triu_indices(mri.shape[0], k=1)
    upper_data = mri.values[upper_indices]

    mri_data = pd.Series(upper_data)
    data = pd.concat([data, mri_data])

    print(data)




if __name__ == '__main__':
    main()
