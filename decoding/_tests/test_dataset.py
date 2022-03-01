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
    builder.bin_responses()
    binned = np.zeros(799)
    binned[40] = binned[200] = 1
    actual_binned = builder._dataset.get_responses()[("events", "neuron_1")].loc[
        ("song_1", 0)
    ]
    assert np.array_equiv(binned, actual_binned)
    builder.add_stimuli()
    builder.create_time_lags()
    dataset = builder.get_dataset()
