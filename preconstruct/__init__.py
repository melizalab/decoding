"""
The `preconstruct` module creates a dataset that is arranged to facilitate
neural decoding, i.e. reconstructing a stimulus given the neural
response that it induces.

To create the dataset, you must create a `preconstruct.sources.DataSource`
and pass it a `preconstruct.dataset.DatasetBuilder` and configure your dataset.

## What this library can do

- Download your [pprox](https://meliza.org/spec:2/pprox/) files
and stimuli in WAVE format from [Neurobank](https://github.com/melizalab/neurobank)
(`preconstruct.sources.NeurobankSource`)
- Allow you to easily switch between different sources of data, such as
local files (`preconstruct.sources.FsSource`)
or Python dictionaries (`preconstruct.sources.MemorySource`)
- Create gammatone-based spectrogram from raw WAVE data of the stimuli
(`preconstruct.dataset.DatasetBuilder.add_stimuli`)
- Automatically update your pprox files to match the
[stimtrial specification](https://github.com/melizalab/lab_specs/blob/main/specs/stimulus_trial.md)
and store the data in a convenient pandas DataFrame
- Convert the point process from the pprox data into an array of fixed duration
bins containing the count of spikes occuring within the corresponding time window
(`preconstruct.dataset.DatasetBuilder.bin_responses`)
- Project the spikes into an alternate basis (`preconstruct.basisfunctions`)
- Rearrange the binned pprox data into windows of a given duration `tau` for
input into a neural decoding model
"""

APP_NAME = "preconstruct"
APP_AUTHOR = "melizalab"
__version__ = "0.1.2"

from preconstruct.dataset import Dataset, DatasetBuilder
from preconstruct import basisfunctions
