import pytest
import numpy as np
from preconstruct.stimuliformats import Gammatone


async def test_margot_data():
    from preconstruct import sources, dataset, basisfunctions
    responses = ["P4_p1r2_ch20_c31", "O129_p1r2_ch19_c3", "P4_p1r2_ch22_c23"]
    stimuli = []
    url = "https://gracula.psyc.virginia.edu/neurobank/"
    test_source = await (sources.NeurobankSource.create(url, stimuli, responses))
    stimuli = list(test_source.stimuli_names_from_pprox())
    data_source = await (sources.NeurobankSource.create(url, stimuli, responses))
    builder = dataset.DatasetBuilder()
    builder.set_data_source(data_source)
    builder.load_responses(ignore_columns=["category"])
    builder.add_stimuli(
        Gammatone(
            window_time=0.0025,
            frequency_bin_count=30,
            min_frequency=500,
            max_frequency=8000,
            log_transform_compress=1,
        ),
        time_step=0.001
    )
    builder.bin_responses()  # 5 ms
    basis = basisfunctions.RaisedCosineBasis(30, linearity_factor=30)
    builder.pool_trials()
    builder.create_time_lags(tau=0.3, basis=basis)
    dataset = builder.get_dataset()
    training_stimuli = stimuli
    print(builder._dataset._get_responses())
    X, Y = dataset[training_stimuli]
    assert np.allclose(
        dataset._get_responses().loc[training_stimuli[0]].index,
        dataset._get_stimuli().loc[training_stimuli[0]].index,
        atol=1e-2,
    )
