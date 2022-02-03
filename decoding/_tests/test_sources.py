from decoding.sources import NeurobankSource, MemorySource

async def test_neurobank():
    stimuli = ['ztqee46x',
            '00oagdl5',
            'g29wxi4q',
            'mrel2o09',
            'vekibwgj',
            'l1a3ltpy',
            'igmi8fxa',
            'c95zqjxq',
            'w08e1crn',
            'jkexyrd5',
            'p1mrfhop',
            ]
    responses = ['P120_1_1_c92']
    url = 'https://gracula.psyc.virginia.edu/neurobank/'
    source = await NeurobankSource.create(url, responses, stimuli)
    assert len(source.get_responses()) == 1
    assert len(source.get_stimuli()) == len(stimuli)

def test_all_formats_equiv(stimuli, cn_pprox, ar_pprox, stimtrial_pprox):
    cn_data = MemorySource(cn_pprox, stimuli)
    ar_data = MemorySource(ar_pprox, stimuli)
    stimtrial = MemorySource(stimtrial_pprox, stimuli)
    print(stimtrial.get_responses())
    print(ar_data.get_responses())
    assert stimtrial == cn_data
    assert stimtrial == ar_data
