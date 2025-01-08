import json

import pandas as pd
import os
import numpy as np
from func import *
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


def main():
    metadata = read_metadata("../data/metadata/training_metadata.csv")
    # show_statistics(metadata)

    metadata = fill_metadata(metadata)
    # show_statistics(metadata)

    # print(metadata)
    maps = mapping_data(metadata)
    #print(metadata)

    with open('../valid_metadata/mappings.json', 'w') as json_file:
        json.dump(maps, json_file, indent=4)
        print("(System): json saved")

    json_file.close()

    # print(metadata)
    # print(maps)

    write_metadata("../valid_metadata/training_metadata_valid.csv", metadata)

    dir_path = "../data/train_tsv"
    list_files = []
    for file in os.listdir(dir_path):
        if file.endswith(".tsv"):
            list_files.append(file)

    model = SGDRegressor(penalty='l1', verbose=1, warm_start=True, tol=0.0001, random_state=10)
    modelR = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=10, random_state=21, verbose=1)
    data_all = []
    data_all_test = []
    data_id = []
    y_all = []

    file_train, file_test = train_test_split(list_files, test_size=0.2, random_state=42)
    err = 0
    train_no = 1

    print("\n\nTraining:")
    for (index, file) in enumerate(file_train):
        if index % 100 == 0:
            print(f"{index}/{len(file_train)}: {file}")
        data, y, id = format_file(file, metadata)
        data_all.append(data[0])
        y_all.append(y[0])
        #model.partial_fit([data[0][-19900:]], y)



    y_test = []
    print("\n\nTesting:")
    for (index, file) in enumerate(file_test):
        if index % 100 == 0:
            print(f"{index}/{len(file_test)}: {file}")
        x, y, id = format_file(file, metadata)
        data_all_test.append(x[0])
        y_test.append(y[0] * 17.0 + 5)
        data_id.append(id)

    data_aux = [lista[-19900:] for lista in data_all]
    model.fit(data_aux, y_all)
    data_aux = [lista[-19900:] for lista in data_all_test]
    y_brain = model.predict(data_aux)
    y_predict_brain = y_brain * 17.0 + 5
    err_brain = mean_squared_error(y_test, y_predict_brain)
    print('Brain - err=', err_brain)

    while train_no == 1 or err > 5:
        print(f"Training no.{train_no}:")
        train_no += 1

        #while err_brain > 4.4:
        #    data_aux = [lista[-19900:] for lista in data_all]
        #    model.fit(data_aux, y_all)
        #    data_aux = [lista[-19900:] for lista in data_all_test]
        #    y_brain = model.predict(data_aux)
        #    y_predict_brain = y_brain * 17.0 + 5
        #    err_brain = mean_squared_error(y_test, y_predict_brain)
        #    print('Brain - err=', err_brain)

        data_aux = [lista[-19900:] for lista in data_all]
        y_brain_tr = model.predict(data_aux)

        data_aux = [lista[:12] for lista in data_all]
        for i in range(len(y_brain_tr)):
            data_aux[i].insert(0, float(y_brain_tr[i]))

        modelR.fit(data_aux, y_all)

        data_aux = [lista[:12] for lista in data_all_test]
        for i in range(len(y_brain)):
            data_aux[i].insert(0, float(y_brain[i]))

        y_predict = modelR.predict(data_aux)
        y_predict = y_predict * 17.0 + 5

        # print(y_test)
        # print(y_predict)

        #result = pd.DataFrame(data, columns=data_cols)
        # print(result)
        #write_metadata('../data/prediction.csv', result)

        err = mean_squared_error(y_test, y_predict)
        print('Brain - err=', err_brain)
        print('Final - err=', err)
        joblib.dump(model, 'models/model.joblib')
        joblib.dump(modelR, 'models/modelR.joblib')


if __name__ == '__main__':
    main()
