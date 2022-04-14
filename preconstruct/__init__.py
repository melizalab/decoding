"""
The `preconstruct` module creates a dataset that is arranged to facilitate
neural decoding, i.e. reconstructing a stimulus given the neural
response that it induces.

To create the dataset, you must create a `preconstruct.sources.DataSource`
and pass it a `preconstruct.dataset.DatasetBuilder` and configure your dataset.

## Examples

See `preconstruct.dataset`.
"""
from joblib import Memory
from appdirs import user_cache_dir

APP_NAME = "preconstruct"
APP_AUTHOR = "melizalab"
__version__ = "0.1.2"


_cache_dir = user_cache_dir(APP_NAME, APP_AUTHOR)
_mem = Memory(_cache_dir, verbose=0)

from preconstruct.dataset import Dataset, DatasetBuilder
from preconstruct import basisfunctions
