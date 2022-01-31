import pytest
import numpy as np
from decoding import DatasetBuilder
from decoding.io import MemorySource

stimtrial_pprox = {
            'neuron_1': {
                "$schema": "https://meliza.org/spec:2/stimtrial.json#",
                'pprox': [
                    {
                        'events': [0.2, 1, 4],
                        'interval': [0, 3],
                        'stimulus': {
                            'interval': [1, 2],
                            'name': 'song_1',
                            }
                        },
                    ]
                }
            }

cn_pprox = {
            'neuron_1': {
                'protocol': 'songs',
                'entry_metadata': {
                    'sampling_rate': 10,
                    },
                'pprox': [
                    {
                        'index': 0,
                        'events': [0.2, 1, 4],
                        'recording': {
                            'start': 0,
                            'stop': 30,
                            },
                        'stim': 'song_1',
                        'stim_on': 1,
                        }
                    ]
                }
            }

ar_pprox = {
            'neuron_1': {
                'experiment': 'induction',
                'pprox': [
                    {
                        'trial': 0,
                        'units': 'ms',
                        'event': [200, 1000, 4000],
                        'stim_uuid': 'song_1',
                        'stim_on': 1,
                        }
                    ]
                }
            }
@pytest.fixture
def stimuli():
    sample_rate = 44100
    samples = np.random.normal(0, 1000, 44100)
    return {
            'song_1': (sample_rate, samples)
            }


@pytest.fixture
def mem_data_source(stimuli):
    responses = stimtrial_pprox
    return MemorySource(responses, stimuli)

def test_all_formats_equiv(stimuli):
    cn_data = MemorySource(cn_pprox, stimuli)
    ar_data = MemorySource(ar_pprox, stimuli)
    stimtrial = MemorySource(stimtrial_pprox, stimuli)
    print(stimtrial.get_responses())
    print(ar_data.get_responses())
    assert stimtrial == cn_data
    assert stimtrial == ar_data

def test_building(mem_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(mem_data_source)
    builder.load_responses()
    builder.bin_responses()
    builder.add_stimuli()
    builder.create_time_lags()
    dataset = builder.get_dataset()
