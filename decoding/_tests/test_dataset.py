import pytest
import numpy as np

from decoding import DatasetBuilder
from decoding.sources import MemorySource


@pytest.fixture
def mem_data_source(stimtrial_pprox, stimuli):
    responses = stimtrial_pprox
    return MemorySource(responses, stimuli)


def test_building(mem_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(mem_data_source)
    builder.load_responses()

    time_step = 0.001
    builder.bin_responses(time_step=time_step)
    neuron = "neuron_1"
    trial_index = 0
    trial = mem_data_source.get_responses()[neuron]["pprox"][trial_index]
    stimulus = trial["stimulus"]
    recording_duration = (trial["interval"][1] - trial["interval"][0]) / time_step
    binned = np.zeros(int(recording_duration))
    binned[200] = binned[1000] = binned[1500] = 1
    actual_binned = builder._dataset.get_responses()[neuron].loc[trial_index]
    assert actual_binned.shape == binned.shape
    assert np.array_equiv(binned, actual_binned)

    builder.add_stimuli()
    print(builder._dataset.get_responses().columns)
    spectrogram = builder._dataset.get_stimuli()["spectrogram"].loc[stimulus["name"]]
    spectrogram_length= spectrogram.shape[0]
    stim_length = builder._dataset.get_stimuli()["stimulus.length"].loc[stimulus["name"]]
    assert spectrogram_length == stim_length

    tau=0.3
    builder.create_time_lags(tau=tau)
    actual_lagged = builder._dataset.get_responses()[neuron].loc[trial_index]
    print(spectrogram)
    shape = (spectrogram_length, int(tau / time_step))
    assert actual_lagged.shape == shape
    # lagged = np.array([0])
    # assert np.array_equiv(actual_lagged, lagged)
    dataset = builder.get_dataset()
    X, Y = dataset[[0]]
    assert len(X) == len(Y)


def test_pool_trials(mem_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(mem_data_source)
    builder.load_responses()
    builder.bin_responses()
    builder.add_stimuli()
    builder.create_time_lags()
    neurons = builder._dataset.get_responses().columns
    builder.pool_trials()
    assert builder._dataset.get_responses().columns == neurons
    dataset = builder.get_dataset()
    X, Y = dataset[['song_1']]
    assert len(X) == len(Y)
