import pytest
import os, json
import pandas as pd

# Fixtures pointing to test files
@pytest.fixture(scope="module")
def examples_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir,"..","examples"))

@pytest.fixture(scope="module")
def binding_curves():

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
        test_data_frames.append(pd.read_csv(f,index_col=1))

    return input_params, test_data_frames






# delete any temporary files
