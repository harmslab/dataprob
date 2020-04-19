import pytest

import likelihood

import pandas as pd
import os, json, pickle

# Fixtures pointing to test files
@pytest.fixture(scope="module")
def examples_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir,"..","examples"))

@pytest.fixture(scope="module")
def binding_curve_test_data():

    def binding_curve(K,X):
        return K*X/(1 + K*X)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.abspath(os.path.join(base_dir,"..","examples"))
    json_file = os.path.join(example_dir,"binding-curves.json")
    json_data = json.load(open(json_file,"r"))

    input_params = json_data["input_params"]

    test_files = json_data["test_files"]
    test_files.sort()

    test_data_frames = []
    for test_file in test_files:
        f = os.path.join(example_dir,test_file)
        test_data_frames.append(pd.read_csv(f,index_col=0))

    return input_params, binding_curve, test_data_frames

@pytest.fixture(scope="module")
def ml_object(binding_curves):

    f = likelihood.MLFitter()

    lm = LikelihoodModel(model,**{"X":df.X})
    f.fit(lm.observable,params,df.Y)

    return f







# delete any temporary files
