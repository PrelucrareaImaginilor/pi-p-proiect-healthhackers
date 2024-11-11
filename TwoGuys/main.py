import pandas as pd
import os
import numpy as np
from func import *

def main():

    metadata = read_metadata("../data/metadata/training_metadata.csv")

    dir_path="../data/train_tsv"
    list_files=[]
    for file in os.listdir(dir_path):
        if file.endswith(".tsv"):
            list_files.append(file)

    print(len(list_files))

    show_statistics(metadata)

    #for (index, file) in enumerate(list_files):
    #    print(index, " ", file)
    #    data = format(file, metadata)




if __name__ == '__main__':
    main()
