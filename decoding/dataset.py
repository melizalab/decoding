"""Loading data for neural decoding

## Examples

Let's build a simple dataset for neural decoding. The first step is determining
what data we'll use. Depending on where our data is stored we'll use different
classes that inherit from decoding.sources.DataSource. For example, if we want to
use data from a local folder on our computer we might use decoding.sources.FsSource.
For other options, see decoding.sources. In this example, we'll use
decoding.sources.NeurobankSource, which will automatically download a list of
identifiers from Neurobank. Let's suppose we know we want to use a pprox responses
file with identifier `P120_1_1_c92`.
>>> from decoding.sources import NeurobankSource
>>> import asyncio
>>> responses = ['P120_1_1_c92']
>>> stimuli = [] # we'll leave this empty for now
>>> url = 'https://gracula.psyc.virginia.edu/neurobank/'
>>> test_source = asyncio.run(NeurobankSource.create(url, stimuli, responses))

We can't use this `DataSource` to build a dataset because it doesn't contain any
of the stimuli that were presented during the recording, but we can easily get
a list of the stimuli identifiers:
>>> stimuli = list(test_source.stimuli_names_from_pprox())
>>> sorted(stimuli)
['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy', 'mrel2o09', \
'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x']

Let's make a new DataSource that includes the stimuli
>>> data_source = asyncio.run(NeurobankSource.create(url, stimuli, responses))

Now we can start building our dataset. We will put our data into a
`DatasetBuilder`, which will allow us to configure the format of our dataset.
>>> from decoding.dataset import DatasetBuilder
>>> builder = DatasetBuilder()
>>> builder.set_data_source(data_source)
>>> builder.load_responses()

The first choice we have to make is the size of our time steps. This will
determine the granularity of the time axis for both the spikes and the stimuli.
The unit for this argument and all other time values will be seconds.
>>> builder.bin_responses(time_step=0.005) # 5 ms

Next, we load the stimuli. We must choose parameters to control how the gammatone
spectrograms are generated. Consult DatasetBuilder.add_stimuli for details on each
argument.
>>> builder.add_stimuli(
...     window_scale=1,
...     frequency_bin_count=50,
...     min_frequency=500,
...     max_frequency=8000,
...     log_transform=True,
...     log_transform_compress=1,
... )

Now we will convert the binned spikes into a lagged matrix, with a with a window
of size tau.
>>> builder.create_time_lags(tau=0.3)

Our next step is to combine data from multiple presentations of the same stimulus.
(This is optional.)
>>> builder.pool_trials()

We have finished building our dataset.
We should investigate our object a bit, to make sure we understand how
it's structured.
>>> dataset = builder.get_dataset()

>>> dataset.index
Index(['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy', 'mrel2o09',
       'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x'],
      dtype='object', name='stim')
>>> dataset.responses.columns
MultiIndex([(  'offset', 'P120_1_1_c92'),
            ('interval', 'P120_1_1_c92'),
            ('stimulus', 'P120_1_1_c92'),
            (  'events', 'P120_1_1_c92')],
           )

Let's use our dataset to perform a simple neural decoding task

>>> from sklearn.linear_model import Ridge
>>> import numpy as np
>>> training_stimuli = ['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy', 'mrel2o09']
>>> X, Y = dataset[training_stimuli]
>>> X.shape, Y.shape
((2476, 60, 1), (2476, 50))
>>> X = np.resize(X, (X.shape[0], X.shape[1] * X.shape[2]))
>>> model = Ridge(alpha=1.0)
>>> model.fit(X, Y)
Ridge()
>>> model.score(X, Y)
0.1903559601102622

"""
from typing import Optional, Callable, Any

import numpy as np
import pandas as pd
from joblib import Memory
from appdirs import user_cache_dir
from gammatone.gtgram import gtgram
from scipy.linalg import hankel
from pandarallel import pandarallel

import decoding
from decoding.sources import DataSource
from decoding.basisfunctions import Basis

pandarallel.initialize()
_cache_dir = user_cache_dir(decoding.APP_NAME, decoding.APP_AUTHOR)
mem = Memory(_cache_dir, verbose=0)


class DatasetBuilder:
    """Construct instances of the `Dataset` class using the [builder
    pattern](https://refactoring.guru/design-patterns/builder)
    """
    data_source: DataSource
    tau: Optional[float]
    basis: Optional[Basis]

    def __init__(self):
        self._dataset = Dataset()
        self.data_source = _EmptySource()

    def set_data_source(self, data_source: DataSource):
        self.data_source = data_source

    def load_responses(self):
        clusters = self.data_source.get_responses()
        assert len(clusters) > 0, "no clusters"
        responses = pd.concat(
            {
                k: pd.DataFrame(v["pprox"]).set_index(["stim", "index"])
                for k, v in clusters.items()
            },
            axis="columns",
        )
        responses.columns = responses.columns.reorder_levels(order=[1, 0])
        self._dataset.responses = responses

    def bin_responses(self, time_step : float = 0.005):
        """
        transform a point process into bins of size `time_step` containinng
        the number of events that occurred within that time bin.
        """
        self._dataset.time_step = time_step
        self._dataset.get_responses()["events"] = self.responses_apply(self._hist)

    def _hist(self, row) -> np.ndarray:
        duration = np.max(row["events"]) if len(row["events"]) else 0
        # we ignore `duration % time_step` at the end
        bin_edges = np.arange(0, duration, self._dataset.get_time_step())
        histogram, _ = np.histogram(row["events"], bin_edges)
        return histogram

    def add_stimuli(
        self,
        window_scale=1,
        frequency_bin_count=50,
        min_frequency=500,
        max_frequency=8000,
        log_transform=True,
        log_transform_compress=1,
    ):
        """
        Add a dataframe containing gammatone spectrograms for each
        stimulus associated with a trial

        `window_scale`: ratio of gammatone window size to time_step
        `log_transform`: whether to take the log of the power of each
        spectrogram. If `True`, each point on the spectrogram `x` will
        be transformed into `log(x + log_transform_compress) - log(x)`
        """
        gammatone_params = {
                'window_time': self._dataset.get_time_step() * window_scale,
                'hop_time': self._dataset.get_time_step(),
                'channels': frequency_bin_count,
                'f_min': min_frequency,
                'f_max': max_frequency,
        }
        if log_transform:
            log_transform_params = {
                    'compress': log_transform_compress,
            }
        else:
            log_transform_params = None
        wav_data = self.data_source.get_stimuli()
        spectrograms = {
            k: self._spectrogram(v, log_transform_params, gammatone_params)
            for k, v in wav_data.items()
        }
        self._dataset.stimuli = (
            self._dataset.get_responses()
            .apply(lambda x: spectrograms[x.name[0]], axis="columns")
            .sort_index()
        )

    @staticmethod
    def _spectrogram(wav_data, log_transform_params, gammatone_params):
        sample_rate, samples = wav_data
        spectrogram = mem.cache(gtgram)(
            samples,
            sample_rate,
            **gammatone_params
        )
        if log_transform_params is not None:
            compress = log_transform_params['compress']
            spectrogram = np.log10(spectrogram + compress) - np.log10(compress)
        return spectrogram.T

    def create_time_lags(self, tau : float = 0.300, basis: Optional[Basis] = None):
        """
        `tau`: length of window (in secs) to consider in prediction
        `basis`: an instance of a class that inherits from
        `decoding.basisfunctions.Basis`, initialized with the dimension
        of the projection

        ## example
        <!--
        >>> import asyncio
        >>> from decoding.sources import NeurobankSource
        >>> responses = ['P120_1_1_c92']
        >>> url = 'https://gracula.psyc.virginia.edu/neurobank/'
        >>> stimuli = ['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy',
        ...         'mrel2o09', 'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x']
        >>> data_source = asyncio.run(NeurobankSource.create(url, stimuli, responses))
        >>> builder = DatasetBuilder()
        >>> builder.set_data_source(data_source)
        >>> builder.load_responses()
        >>> builder.bin_responses(time_step=0.005) # 5 ms
        >>> builder.add_stimuli(
        ...     window_scale=1,
        ...     frequency_bin_count=50,
        ...     min_frequency=500,
        ...     max_frequency=8000,
        ...     log_transform=True,
        ...     log_transform_compress=1,
        ... )


        -->
        >>> from decoding.basisfunctions import RaisedCosineBasis
        >>> builder.create_time_lags(tau=0.3, basis=RaisedCosineBasis(30))


        """

        self.tau = tau
        self.basis = basis
        self._dataset.get_responses()["events"] = self.responses_apply(self._stagger)

    def _stagger(self, row):
        start = self._dataset.to_steps(row["stimulus"]["interval"][0])
        window_length = self._dataset.to_steps(self.tau)
        stop = start + self._dataset.get_stimuli()[row.name].shape[0]
        total_time_steps = start + stop - 1 + window_length
        pad_width = max(0, total_time_steps - len(row["events"]))
        events = np.pad(row["events"], (0, pad_width))
        assert len(events) >= total_time_steps
        time_lagged = hankel(
            events[start:stop], events[stop - 1 : stop - 1 + window_length]
        )
        if self.basis is not None:
            basis_matrix = self.basis.get_basis(window_length)
            return np.dot(time_lagged, basis_matrix)
        return time_lagged

    def pool_trials(self):
        """Pool spikes across trials"""
        neurons = self._dataset.get_responses()['events'].columns
        events = pd.concat(
                {"events": self._dataset.get_responses()["events"].groupby("stim").agg({n: 'sum' for n in neurons})},
            axis="columns",
        )
        # we assume that all fields except for events are the same across trials
        # because we don't have a way to aggregate other datatypes
        self._dataset.responses = (
            self._dataset.get_responses()
            .groupby("stim")
            .first()
            .drop("events", axis="columns", level=0)
            .join(events)
        )
        self._dataset.stimuli = self._dataset.get_stimuli().groupby("stim").first()

    def responses_apply(self, func: Callable[[pd.DataFrame], Any]):
        """
        Responses has a complex structure; this function provides a
        simple way to apply a function to each row

        `func`: (row of responses dataframe) -> (element of output series)

        returns (pandas.Series): the collected outputs of `func`
        """
        return (
            self._dataset.get_responses()
            .groupby(level=1, axis="columns")
            .parallel_apply(
                lambda x: x.droplevel(1, axis="columns").apply(func, axis="columns")
            )
        )

    def get_dataset(self):
        """Return the fully constructed `Dataset` object"""
        dataset = self._dataset
        dataset.responses = dataset.get_responses().sort_index()
        assert np.array_equal(
            dataset.get_responses().index, dataset.get_stimuli().index
        )
        dataset.index = dataset.get_responses().index
        return dataset


class Dataset:
    """Holds constructed response matrix and stimuli
    """
    responses: Optional[pd.DataFrame]
    """"""
    stimuli: Optional[pd.DataFrame]
    """"""
    time_step: Optional[float]
    """granularity of time"""
    index: Optional[pd.Index]
    """"""

    def __init__(self):
        pass

    def get_stimuli(self) -> pd.DataFrame:
        if self.stimuli is None:
            raise InvalidConstructionSequence("must call `add_stimuli` first")
        return self.stimuli

    def get_time_step(self) -> float:
        if self.time_step is None:
            raise InvalidConstructionSequence("must call `bin_responses` first")
        return self.time_step

    def get_responses(self) -> pd.DataFrame:
        if self.responses is None:
            raise InvalidConstructionSequence("must call `load_responses` first")
        return self.responses

    def __getitem__(self, key):
        """
        get numpy arrays representing the responses and the stimuli
        at the given pandas index range
        """
        events = self.get_responses().loc[key]["events"]
        responses = np.concatenate(
            [np.stack(x, axis=2) for x in events.values.tolist()]
        )
        stimuli = np.concatenate(self.get_stimuli().loc[key].values)
        return responses, stimuli

    def to_steps(self, time_in_seconds):
        """Converts a time in seconds to a time in steps"""
        return int(time_in_seconds / self.get_time_step())


class _EmptySource(DataSource):
    def _get_raw_responses(self):
        self._raise()

    def get_stimuli(self):
        self._raise()

    @staticmethod
    def _raise():
        raise InvalidConstructionSequence(
            "Must call DatasetBuilder.set_data_source"
            "before using methods that use data"
        )


class InvalidConstructionSequence(Exception):
    """Indicates that the methods of a DatasetBuilder have been called in an invalid order"""

    def __init__(self, description):
        super().__init__()
        self.description = description

    def __str__(self):
        return f"invalid construction sequence: {self.description}"
