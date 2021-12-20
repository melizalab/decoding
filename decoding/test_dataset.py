import pytest
import numpy as np
from decoding import DatasetBuilder
from decoding.io import MemorySource

stimtrial = {
            'neuron_1': {
                "$schema": "https://meliza.org/spec:2/stimtrial.json#",
                'pprox': [
                    {
                        'events': [0.2, 1, 4],
                        'interval': [0, 5],
                        'stimulus': {
                            'interval': [1, 2],
                            'name': 'song_1',
                            }
                        },
                    ]
                }
            }

sam_pprox = {
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
                            'stop': 50,
                            },
                        'stim': 'song_1',
                        'stim_on': 1,
                        }
                    ]
                }
            }

@pytest.fixture(params = [stimtrial, sam_pprox])
def mem_data_source(request):
    responses = request.param
    sample_rate = 44100
    samples = np.random.normal(0, 1000, 86000)
    stimuli = {
            'song_1': (sample_rate, samples)
            }
    return MemorySource(responses, stimuli)

def test_building(mem_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(mem_data_source)
    builder.load_responses()
    builder.bin_responses()
    builder.add_stimuli()
    builder.create_time_lags()
    dataset = builder.get_dataset()
