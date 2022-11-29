# -*- coding: utf-8 -*-
# -*- mode: python -*-
import pytest
import numpy as np
from preconstruct import DatasetBuilder
from preconstruct.sources import MemorySource
from preconstruct.stimuliformats import Spectrogram, Gammatone

def test_spectrogram(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    min_frequency = 1000
    max_frequency = 8000
    builder.add_stimuli(
        Spectrogram(
            scaling="density",
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            log_transform_compress=1,
        ),
        time_step=0.005
    )
    dataset = builder.get_dataset()
    frequency_bands = dataset._get_stimuli().columns
    assert frequency_bands[0] - min_frequency > -1e-9
    assert max_frequency - frequency_bands[-1] > -1e-9


def test_gammatone(real_data_source):
    time_step = 0.005
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    min_frequency = 1000
    max_frequency = 8000
    frequency_bin_count = 50
    builder.add_stimuli(
        Gammatone(
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            frequency_bin_count=frequency_bin_count
        ), 
        time_step=time_step
    )
    dataset = builder.get_dataset()
    frequency_bands = dataset._get_stimuli().columns
    assert frequency_bands[0] - min_frequency > -1e-9
    assert max_frequency - frequency_bands[-1] > -1e-9


def test_syllable(real_data_source):
    from preconstruct.stimuliformats import SyllableCategorical
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.add_stimuli(SyllableCategorical(), time_step=0.005)
    builder.bin_responses()
    builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert X.shape[0] == Y.shape[0]
    trial = dataset._get_trial_data().iloc[0]
    assert np.allclose(
        dataset._get_responses().loc[trial.name].index,
        dataset._get_stimuli().loc[trial['stimulus.name']].index,
        atol=1e-6,
    )


@pytest.mark.parametrize("format", [Gammatone, Spectrogram])
def test_known_spectrogram(known_spectrogram_source, format):
    # make sure that the frequency band with the most power is the one that
    # is closest to 3kHz for a signal that is a sine wave oscillating at 3kHz
    builder = DatasetBuilder()
    builder.set_data_source(known_spectrogram_source)
    builder.load_responses()
    builder.add_stimuli(format(), time_step=0.005)
    dataset = builder.get_dataset()
    spec = dataset._get_stimuli().loc["song_1"]
    freq_band = 3e3
    closest_column = (spec.columns.to_series() - freq_band).abs().argmin()
    assert spec.mean().argmax() == closest_column
