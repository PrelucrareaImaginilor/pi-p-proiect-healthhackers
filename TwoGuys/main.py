import pandas as pd
import os
import numpy as np
from func import *

def main():

    metadata = read_metadata("../data/metadata/training_metadata.csv")
    show_statistics(metadata)

    metadata = fill_metadata(metadata)
    show_statistics(metadata)

    write_metadata("../valid_metadata/training_metadata_valid.csv", metadata)

    dir_path="../data/train_tsv"
    list_files=[]
    for file in os.listdir(dir_path):
        if file.endswith(".tsv"):
            list_files.append(file)


    if 0:
        for (index, file) in enumerate(list_files):
            print(f"{index}/{len(list_files)}: {file}")
            data = format(file, metadata)




if __name__ == '__main__':
    main()
