import pytest
from decoding.sources import NeurobankSource, MemorySource

@pytest.fixture
def nbank_stimuli():
    return [
        "ztqee46x",
        "g29wxi4q",
        "mrel2o09",
        "vekibwgj",
        "l1a3ltpy",
        "igmi8fxa",
        "c95zqjxq",
        "w08e1crn",
        "jkexyrd5",
        "p1mrfhop",
    ]

@pytest.fixture
def nbank_responses():
    return ["P120_1_1_c92"]

@pytest.fixture
def nbank_url():
    return "https://gracula.psyc.virginia.edu/neurobank/"

async def test_neurobank(nbank_responses, nbank_stimuli, nbank_url):
    source = await NeurobankSource.create(nbank_url, nbank_stimuli, nbank_responses)
    assert len(source.get_responses()) == 1
    assert len(source.get_stimuli()) == len(nbank_stimuli)


def test_all_formats_equiv(stimuli, cn_pprox, ar_pprox, stimtrial_pprox):
    cn_data = MemorySource(cn_pprox, stimuli)
    ar_data = MemorySource(ar_pprox, stimuli)
    stimtrial = MemorySource(stimtrial_pprox, stimuli)
    print(stimtrial.get_responses())
    print(ar_data.get_responses())
    assert stimtrial == cn_data
    assert stimtrial == ar_data

async def test_show_stimuli(nbank_responses, nbank_stimuli, nbank_url):
    source = await NeurobankSource.create(nbank_url, nbank_stimuli, nbank_responses)
    assert source.show_stimuli() == set(nbank_stimuli)
