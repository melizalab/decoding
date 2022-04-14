import pytest
import pandas as pd
from preconstruct import DatasetBuilder
from preconstruct.dataset import IncompatibleStimuliFormat
from preconstruct.stimuliformats import *

def test_categorical(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.bin_responses()
    builder.add_stimuli(Categorical())
    stimuli = pd.Series(['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy',
       'mrel2o09', 'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x'])
    actual_stimuli = builder._dataset.get_stimuli()["stimulus"].reset_index(drop=True)
    assert (actual_stimuli == stimuli).all()
    with pytest.raises(IncompatibleStimuliFormat):
        builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert Y.shape == (100, 10)
    assert np.equal(np.sum(Y, axis=1), np.ones(100)).all()
    all_time_steps_across_all_trials = 91178
    n_neurons = 2
    dummy = 1
    assert X.shape == (all_time_steps_across_all_trials, dummy, n_neurons)
