"""
The `decoding` module creates a dataset that is arranged to facilitate
neural decoding, i.e. reconstructing a stimulus given the neural
response that it induces.

To create the dataset, you must create a `decoding.sources.DataSource`
and pass it a `decoding.dataset.DatasetBuilder` and configure your dataset.

## Examples

See `decoding.dataset`.
"""

APP_NAME = "decoding"
APP_AUTHOR = "melizalab"

from decoding.dataset import Dataset, DatasetBuilder
from decoding import basisfunctions
