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

    test_file = json_data["test_file"]

    f = os.path.join(example_dir,test_file)
    test_df = pd.read_csv(f,index_col=0)

    return input_params, binding_curve, test_df

@pytest.fixture(scope="module")
def fit_tolerance_fixture():
    return 0.1







# delete any temporary files
