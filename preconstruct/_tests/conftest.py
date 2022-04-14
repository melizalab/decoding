import pytest
import numpy as np
from preconstruct.sources import NeurobankSource, MemorySource


@pytest.fixture
def stimtrial_pprox():
    return {
        "neuron_1": {
            "$schema": "https://meliza.org/spec:2/stimtrial.json#",
            "pprox": [
                {
                    "events": [0.2, 1, 1.5],
                    "interval": [0, 3],
                    "stimulus": {
                        "interval": [1, 2],
                        "name": "song_1",
                    },
                },
            ],
        }
    }


@pytest.fixture
def cn_pprox():
    return {
        "neuron_1": {
            "protocol": "songs",
            "entry_metadata": {
                "sampling_rate": 10,
            },
            "pprox": [
                {
                    "index": 0,
                    "events": [0.2, 1, 1.5],
                    "recording": {
                        "start": 0,
                        "stop": 30,
                    },
                    "stim": "song_1",
                    "stim_on": 1,
                }
            ],
        }
    }


@pytest.fixture
def ar_pprox():
    return {
        "neuron_1": {
            "experiment": "induction",
            "pprox": [
                {
                    "trial": 0,
                    "units": "ms",
                    "event": [200, 1000, 1500],
                    "stim_uuid": "song_1",
                    "stim_on": 1,
                }
            ],
        }
    }


@pytest.fixture
def stimuli():
    sample_rate = 44100
    samples = np.random.normal(0, 1000, 44100)
    return {"song_1": (sample_rate, samples)}

@pytest.fixture
def mem_data_source(stimtrial_pprox, stimuli):
    responses = stimtrial_pprox
    return MemorySource(responses, stimuli)

@pytest.fixture
async def real_data_source():
    responses = ['P120_1_1_c92', 'P120_1_1_c89']
    stimuli = \
    ['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy', 'mrel2o09', \
    'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x']
    url = 'https://gracula.psyc.virginia.edu/neurobank/'
    return await NeurobankSource.create(url, stimuli, responses)
