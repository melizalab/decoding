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

def test_pool_trials_before_lag(mem_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(mem_data_source)
    builder.load_responses()
    builder.bin_responses()
    builder.add_stimuli()
    builder.pool_trials()
    builder.create_time_lags()
    neurons = builder._dataset.get_responses().columns
    assert builder._dataset.get_responses().columns == neurons
    dataset = builder.get_dataset()
    X, Y = dataset[['song_1']]
    assert len(X) == len(Y)

async def test_margot_data():
    from decoding import sources, dataset, basisfunctions
    from sklearn.linear_model import RidgeCV, Ridge
    responses = ['P4_p1r2_ch20_c31','O129_p1r2_ch19_c3','P4_p1r2_ch22_c23']
#responses = "unitfiles_l2a.txt"
    stimuli = [] # we'll leave this empty for now
    url = 'https://gracula.psyc.virginia.edu/neurobank/'
    test_source = await (sources.NeurobankSource.create(url, stimuli, responses))
    stimuli = list(test_source.stimuli_names_from_pprox())
    data_source = await (sources.NeurobankSource.create(url, stimuli, responses))
    builder = dataset.DatasetBuilder()
    builder.set_data_source(data_source)
    print("loading responses")
    builder.load_responses(ignore_columns=["category"])
    print("binning responses")
    builder.bin_responses(time_step=0.001) # 5 ms
    print("computing spectrograms")
    builder.add_stimuli(
         window_scale=2.5,
         frequency_bin_count=30,
         min_frequency=500,
         max_frequency=8000,
         log_transform=True,
         log_transform_compress=1,
    )
    basis = basisfunctions.RaisedCosineBasis(30, linearity_factor=30)
    print("creating design matrices")
    builder.pool_trials()
    builder.create_time_lags(tau=0.3, basis=basis)
    dataset = builder.get_dataset()
    training_stimuli = stimuli
    print(builder._dataset.responses)
    X, Y = dataset[training_stimuli]
    X = np.resize(X, (X.shape[0], X.shape[1] * X.shape[2]))
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    estimator = Ridge(alpha=8.59)
    estimator.fit(X, Y)
    print("model score:", estimator.score(X, Y))
