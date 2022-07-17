from preconstruct import DatasetBuilder
from preconstruct.stimuliformats import *


def test_spectrogram(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.bin_responses()
    min_frequency = 1000
    max_frequency = 8000
    builder.add_stimuli(
        Spectrogram(
            scaling="density", min_frequency=min_frequency, max_frequency=max_frequency
        )
    )
    builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert X.shape[0] == Y.shape[0]
    frequency_bands = dataset.get_stimuli().columns
    assert ((frequency_bands > min_frequency) & (frequency_bands < max_frequency)).all()


def test_gammatone(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.bin_responses()
    frequency_bin_count = 50
    builder.add_stimuli(Gammatone(frequency_bin_count=frequency_bin_count))
    builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert Y.shape[1] == frequency_bin_count
    assert X.shape[0] == Y.shape[0]
