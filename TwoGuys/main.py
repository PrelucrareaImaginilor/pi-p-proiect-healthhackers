import json

import pandas as pd
import os
import numpy as np
from func import *
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def main():

    metadata = read_metadata("../data/metadata/training_metadata.csv")
    #show_statistics(metadata)

    metadata = fill_metadata(metadata)
    #show_statistics(metadata)

    #print(metadata)
    maps = mapping_data(metadata)

    with open('../valid_metadata/mappings.json', 'w') as json_file:
        json.dump(maps, json_file, indent=4)
        print("(System): json saved")

    json_file.close()

    #print(metadata)
    #print(maps)

    write_metadata("../valid_metadata/training_metadata_valid.csv", metadata)

    dir_path="../data/train_tsv"
    list_files=[]
    for file in os.listdir(dir_path):
        if file.endswith(".tsv"):
            list_files.append(file)

    #model = SGDRegressor(penalty='l1', warm_start=True)
    model = MLPRegressor(hidden_layer_sizes=(10,), warm_start=True)

    file_train, file_test = train_test_split(list_files, test_size=0.2, random_state=42)
    err = 0
    train_no = 1
    while train_no == 1 or err > 3:
        print(f"Training no.{train_no}:")
        train_no += 1
        for (index, file) in enumerate(file_train):
            if index % 100 == 0:
                print(f"{index}/{len(file_train)}: {file}")
            data, y, id = format(file, metadata)
            model.partial_fit(data, y)



        y_test = []
        y_predict = []
        data_cols = ['participant_id', 'age']
        data = []
        result = []
        print("\n\nTesting:")
        for (index, file) in enumerate(file_test):
            if index % 100 == 0:
                print(f"{index}/{len(file_test)}: {file}")
            x, y, id = format(file, metadata)
            yo = model.predict(x)
            yo = np.clip(yo, 5, 22)
            y_test.append(y[0])
            y_predict.append(yo[0])
            data.append([id, round(float(yo[0]), 2)])

        #print(y_test)
        #print(y_predict)

        result = pd.DataFrame(data, columns=data_cols)
        #print(result)
        write_metadata('../data/prediction.csv', result)


        err = mean_squared_error(y_test, y_predict)
        print('err=', err)
        joblib.dump(model, 'models/test.joblib')


if __name__ == '__main__':
    main()
