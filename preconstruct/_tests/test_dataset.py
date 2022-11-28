import pytest
import numpy as np
import pandas as pd

from preconstruct import DatasetBuilder
from preconstruct.sources import NeurobankSource, MemorySource
from preconstruct.stimuliformats import Gammatone
from preconstruct.dataset import InvalidConstructionSequence, TimestepSetTwice

@pytest.fixture
def mem_data_source(stimtrial_pprox, stimuli):
    responses = stimtrial_pprox
    return MemorySource(responses, stimuli)


@pytest.fixture
async def real_data_source():
    responses = ["P120_1_1_c92", "P120_1_1_c89"]
    stimuli = [
        "c95zqjxq",
        "g29wxi4q",
        "igmi8fxa",
        "jkexyrd5",
        "l1a3ltpy",
        "mrel2o09",
        "p1mrfhop",
        "vekibwgj",
        "w08e1crn",
        "ztqee46x",
    ]
    url = "https://gracula.psyc.virginia.edu/neurobank/"
    return await NeurobankSource.create(url, stimuli, responses)


def test_building(mem_data_source):
    time_step = 0.001
    builder = DatasetBuilder()
    builder.set_data_source(mem_data_source)
    builder.load_responses()
    neuron = "neuron_1"
    trial_index = 0
    trial = mem_data_source.get_responses()[neuron]["pprox"][trial_index]
    stimulus = trial["stimulus"]["name"]

    builder.add_stimuli(Gammatone(), time_step=time_step)
    spectrogram = builder._dataset._get_stimuli().loc[stimulus]
    spectrogram_length = spectrogram.shape[0]

    builder.bin_responses(normalize=False)
    binned = builder._dataset._get_responses()[neuron].loc[trial_index]
    bin_times = binned[binned > 0].index.to_numpy()
    np.testing.assert_array_almost_equal(bin_times, trial["events"], decimal=3)

    # check for misalignment of stimulus and response bins:
    resp_bins = binned.index
    stim_bins = spectrogram.index
    assert np.diff(resp_bins).mean() == pytest.approx(np.diff(stim_bins).mean())

    tau = 0.3
    builder.create_time_lags(tau=tau)
    actual_lagged = builder._dataset._get_responses()[neuron].loc[trial_index]
    shape = (spectrogram_length, int(tau / time_step))
    assert actual_lagged.shape == shape
    dataset = builder.get_dataset()
    X, Y = dataset[[0]]
    assert len(X) == len(Y)


def test_pool_before_binning(mem_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(mem_data_source)
    builder.load_responses()
    with pytest.raises(InvalidConstructionSequence):
        builder.pool_trials()


def test_pool_trials(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.add_stimuli(Gammatone(), time_step=0.005)
    builder.bin_responses()
    builder.pool_trials()

    trial_index = 0
    neuron = real_data_source.cluster_list[0]
    trial = real_data_source.get_responses()[neuron]["pprox"][trial_index]
    stimulus = trial["stimulus"]["name"]
    binned = builder._dataset._get_responses()[neuron].loc[stimulus]
    spectrogram = builder._dataset._get_stimuli().loc[stimulus]
    assert (
        np.diff(binned.index).mean() ==
        pytest.approx(np.diff(spectrogram.index).mean())
    )

    builder.create_time_lags()
    neurons = builder._dataset._get_responses().columns
    assert (builder._dataset._get_responses().columns == neurons).all()
    dataset = builder.get_dataset()
    X, Y = dataset[["ztqee46x"]]
    assert len(X) == len(Y)
    assert isinstance(dataset._get_responses().columns, pd.MultiIndex)


def test_pool_trials_before_lag(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.add_stimuli(Gammatone(), time_step=0.005)
    builder.bin_responses()
    builder.pool_trials()
    builder.create_time_lags()
    dataset_pool_first = builder.get_dataset()
    # compare with pooling after lag
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.add_stimuli(Gammatone(), time_step=0.005)
    builder.bin_responses()
    builder.create_time_lags()
    builder.pool_trials()
    dataset_pool_second = builder.get_dataset()
    assert np.all(
        [
            np.array_equal(a, b)
            for a, b in zip(dataset_pool_first[:], dataset_pool_second[:])
        ]
    )


def test_datasource_has_diff_stimuli(stimtrial_pprox):
    frequency_bin_count = 50
    sample_rate = 44100
    samples = np.random.normal(0, 1000, 44100)
    # note that the stimuli dict contains a strict superset of the stimuli referenced
    # in the responses dict. That's the point of this test
    stimuli = {"song_1": (sample_rate, samples), "song_2": (sample_rate, samples)}
    responses = stimtrial_pprox
    builder = DatasetBuilder()
    builder.set_data_source(MemorySource(responses, stimuli))
    builder.load_responses()
    builder.add_stimuli(Gammatone(frequency_bin_count=frequency_bin_count), time_step=0.005)
    builder.bin_responses()
    builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert Y.shape[1] == frequency_bin_count
    assert X.shape[0] == Y.shape[0]

def test_stimuli_only(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    frequency_bin_count = 50
    builder.add_stimuli(
        Gammatone(frequency_bin_count=frequency_bin_count), time_step=0.05
    )
    dataset = builder.get_dataset()
    Y = dataset._get_stimuli()
    assert Y.shape[1] == frequency_bin_count
