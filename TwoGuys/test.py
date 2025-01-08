import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from func import read_metadata, fill_metadata, mapping_data, write_metadata, format_file


def main():
    metadata = read_metadata("../data/metadata/test_metadata.csv")
    metadata = fill_metadata(metadata)
    maps = mapping_data(metadata)

    dir_path = "../data/test_tsv"
    list_files = []
    for file in os.listdir(dir_path):
        if file.endswith(".tsv"):
            list_files.append(file)

    model = joblib.load('models/model.joblib')
    modelR = joblib.load('models/modelR.joblib')

    y_predict = []
    data_cols = ['participant_id', 'age']
    data = []
    result = []
    print("\n\nTesting:")
    for (index, file) in enumerate(list_files):
        if index % 100 == 0:
            print(f"{index}/{len(list_files)}: {file}")
        x, y, id = format_file(file, metadata, "test_tsv/")
        yo = model.predict([x[0][-19900:]])
        yo = modelR.predict([[yo[0]] + x[0][:12]])
        yo = yo * 17.0 + 5
        y_predict.append(yo[0])
        data.append([id, round(float(yo[0]), 2)])

    # print(y_test)
    # print(y_predict)

    result = pd.DataFrame(data, columns=data_cols)
    print(result)
    write_metadata('../data/prediction.csv', result)


if __name__ == '__main__':
    main()