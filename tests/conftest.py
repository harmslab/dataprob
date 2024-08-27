import pytest

import pandas as pd

import os
import json
import glob

def get_files(base_dir):
    """
    Traverse base_dir and return a dictionary that keys all files and some
    rudimentary *.ext expressions to absolute paths to those files. They keys
    will be things like "some_dir/test0/rocket.txt" mapping to
    "c:/some_dir/life/base_dir/some_dir/test0/rocket.txt". The idea is to have
    easy-to-read cross-platform keys within unit tests.

    Classes of keys:

        + some_dir/test0/rocket.txt maps to a file (str)
        + some_dir/test0/ maps to the test0 directory itself (str)
        + some_dir/test0/*.txt maps to all .txt (list)
        + some_dir/test0/* maps to all files or directories in the directory
          (list)

    Note that base_dir is *not* included in the keys. All are relative to that
    directory by :code:`os.path.basename(__file__)/{base_dir}`.

    Parameters
    ----------
    base_dir : str
        base directory for search. should be relative to test file location.

    Returns
    -------
    output : dict
        dictionary keying string paths to absolute paths
    """

    containing_dir = os.path.dirname(os.path.realpath(__file__))
    starting_dir = os.path.abspath(os.path.join(containing_dir,base_dir))

    base_length = len(starting_dir.split(os.sep))

    # Traverse starting_dir
    output = {}
    for root, dirs, files in os.walk(starting_dir):

        # path relative to base_dir as a list
        this_path = root.split(os.sep)[base_length:]

        # Build paths to specific files
        local_files = []
        for file in files:
            local_files.append(os.path.join(root,file))
            new_key = this_path[:]
            new_key.append(file)
            output["/".join(new_key)] = local_files[-1]

        # Build paths to patterns of file types
        patterns = {}
        ext = list(set([f.split(".")[-1] for f in local_files]))
        for e in ext:
            new_key = this_path[:]
            new_key.append(f"*.{e}")
            output["/".join(new_key)] = glob.glob(os.path.join(root,f"*.{e}"))

        # Build path to all files in this directory
        new_key = this_path[:]
        new_key.append("*")
        output["/".join(new_key)] = glob.glob(os.path.join(root,f"*"))

        # Build paths to directories in this directory
        for this_dir in dirs:
            new_key = this_path[:]
            new_key.append(this_dir)
            # dir without terminating /
            output["/".join(new_key)] = os.path.join(root,this_dir)

            # dir with terminating /
            new_key.append("")
            output["/".join(new_key)] = os.path.join(root,this_dir)

    # make sure output is sorted stably
    for k in output:
        if issubclass(type(output[k]),str):
            continue

        new_output = list(output[k])
        new_output.sort()
        output[k] = new_output

    return output

@pytest.fixture(scope="module")
def spreadsheets():
    
    files = get_files(os.path.join("test_data","spreadsheets"))
    
    return files

@pytest.fixture(scope="module")
def binding_curve_test_data():
    """
    Main set of binding test data for testing fits.
    """

    # Find directory with test files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.abspath(os.path.join(base_dir,"test_data","binding_curve"))

    # Load json describing test informations
    json_file = os.path.join(example_dir,"binding-curves.json")
    json_data = json.load(open(json_file,"r"))

    # Load csv with fit edata
    test_file = json_data["test_file"]
    f = os.path.join(example_dir,test_file)
    json_data["df"] = pd.read_csv(f,index_col=0)

    # ------------------------------------------------------------
    # Save a model that should be readily wrapped by ModelWrapper
    # ------------------------------------------------------------

    def wrappable_model(K=1,df=None):
        """
        A form of the model that should be wrappable by ModelWrapper.
        """
        return K*df.x/(1 + K*df.x)

    json_data["wrappable_model"] = wrappable_model


    return json_data

@pytest.fixture(scope="module")
def fit_tolerance_fixture():
    """
    Fit tolerance for checking (relative tolerance)
    """
    return 0.1

## Code for skipping slow tests.
def pytest_addoption(parser):
    parser.addoption("--runslow",
                     action="store_true",
                     default=False,
                     help="run slow tests")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
