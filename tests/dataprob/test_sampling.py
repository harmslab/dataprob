import pytest

import dataprob

import numpy as np

import os

def test_init():

    f = dataprob.MLFitter()
    assert f.fit_type == "maximum dataprob"

def test_write_append_samples(binding_curve_test_data,tmp_path):
    """
    Test the ability to read and append samples.
    """

    out_pickle1 = os.path.join(tmp_path,"test1.pickle")
    out_pickle2 = os.path.join(tmp_path,"test2.pickle")

    f = dataprob.MLFitter()
    model = binding_curve_test_data["prewrapped_model"]
    guesses = binding_curve_test_data["guesses"]
    df = binding_curve_test_data["df"]
    input_params = np.array(binding_curve_test_data["input_params"])

    # Should not write because no samples yet
    f.write_samples(out_pickle1)
    assert not os.path.isfile(out_pickle1)

    f.fit(model=model,guesses=guesses,y_obs=df.Y,y_stdev=df.Y_stdev)

    # Make sure fit worked
    assert f.success

    # Make sure the sammples look right
    assert np.array_equal(f.samples.shape,(f._num_samples,1))

    # Now it should have been written out.
    f.write_samples(out_pickle1)
    assert os.path.isfile(out_pickle1)

    # Append

    # Should do anything
    assert f.append_samples() is None

    # Not an array
    with pytest.raises(ValueError):
        f.append_samples(sample_array="test")

    # Not an array
    with pytest.raises(ValueError):
        f.append_samples(sample_array=1)

    # array of wrong dimensions
    with pytest.raises(ValueError):
        f.append_samples(sample_array=np.eye(3,dtype=float))

    # array of right dimensons, wrong type
    with pytest.raises(ValueError):
        f.append_samples(sample_array=np.eye(1,dtype=int))

    # Can't specify array and input file
    with pytest.raises(ValueError):
        f.append_samples(sample_array=np.eye(1),sample_file=out_pickle1)

    # Load an input array as an array
    current_size = f.samples.shape[0]
    with pytest.warns(UserWarning):
        f.append_samples(sample_array=f.samples)
    assert np.array_equal(f.samples.shape,[2*current_size,1])

    # Load an input array from a pickle
    with pytest.warns(UserWarning):
        f.append_samples(sample_file=out_pickle1)
    assert np.array_equal(f.samples.shape,[3*current_size,1])

    # Now try to write to existing file
    with pytest.raises(FileExistsError):
        f.write_samples(out_pickle1)

    # Now write correctly
    f.write_samples(out_pickle2)
    assert os.path.isfile(out_pickle2)

    # Load the written out file to make sure it worked
    with pytest.warns(UserWarning):
        f.append_samples(sample_file=out_pickle2)
    assert np.array_equal(f.samples.shape,[6*current_size,1])

    # Create a new fitter and load samples into an empty fitter (won't work)
    f = dataprob.MLFitter()
    with pytest.raises(ValueError):
        f.append_samples(sample_file=out_pickle1)
