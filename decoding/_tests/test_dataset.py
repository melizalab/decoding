import pytest
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
    builder.add_stimuli()
    builder.create_time_lags()
    dataset = builder.get_dataset()
