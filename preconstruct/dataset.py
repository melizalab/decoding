"""Loading data for neural decoding

## Examples

Let's build a simple dataset for neural decoding. The first step is determining
what data we'll use. Depending on where our data is stored we'll use different
classes that inherit from preconstruct.sources.DataSource. For example, if we want to
use data from a local folder on our computer we might use preconstruct.sources.FsSource.
For other options, see preconstruct.sources. In this example, we'll use
preconstruct.sources.NeurobankSource, which will automatically download a list of
identifiers from Neurobank. Let's suppose we know we want to use a pprox responses
from two files with identifiers `P120_1_1_c92` and `P120_1_1_c89`.
>>> from preconstruct.sources import NeurobankSource
>>> import asyncio
>>> responses = ['P120_1_1_c92', 'P120_1_1_c89']
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
>>> from preconstruct.dataset import DatasetBuilder
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
>>> from preconstruct.stimuliformats import Gammatone
>>> builder.add_stimuli(
...     Gammatone(
...         window_time=0.005,
...         frequency_bin_count=50,
...         min_frequency=500,
...         max_frequency=8000,
...         log_transform_compress=1,
...     )
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

>>> dataset.responses.index
Index(['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy', 'mrel2o09',
       'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x'],
      dtype='object', name='stimulus.name')
>>> dataset.responses.columns
Index(['P120_1_1_c92', 'P120_1_1_c89'], dtype='object')

Let's use our dataset to perform a simple neural decoding task

>>> from sklearn.linear_model import Ridge
>>> import numpy as np
>>> training_stimuli = ['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy', 'mrel2o09']
>>> test_stimuli = set(dataset.responses.index).difference(training_stimuli)
>>> X, Y = dataset[list(training_stimuli)]
>>> X.shape, Y.shape
((2476, 60, 2), (2476, 50))
>>> X = np.resize(X, (X.shape[0], X.shape[1] * X.shape[2]))
>>> model = Ridge(alpha=1.0)
>>> model.fit(X, Y)
Ridge()
>>> model.score(X, Y)
0.30159992
>>> X_test, Y_test = dataset[list(test_stimuli)]
>>> X_test = np.resize(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
>>> model.score(X_test, Y_test)
0.15341238

"""
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import hankel

from preconstruct.sources import DataSource
from preconstruct.basisfunctions import Basis
from preconstruct.stimuliformats import StimuliFormat


class _IncompleteDataset:
    responses: Optional[pd.DataFrame]
    stimuli: Optional[pd.DataFrame]
    trial_data: Optional[pd.DataFrame]
    time_step: Optional[float]
    stimuli_format: Optional[StimuliFormat]

    def get_stimuli(self) -> pd.DataFrame:
        if self.stimuli is None:
            raise InvalidConstructionSequence("add_stimuli")
        return self.stimuli

    def get_time_step(self) -> float:
        if self.time_step is None:
            raise InvalidConstructionSequence("bin_responses")
        return self.time_step

    def get_trial_data(self) -> pd.DataFrame:
        if self.trial_data is None:
            raise InvalidConstructionSequence("load_responses")
        return self.trial_data

    def get_responses(self) -> pd.DataFrame:
        if self.responses is None:
            raise InvalidConstructionSequence("load_responses")
        return self.responses

    def get_stimuli_format(self) -> StimuliFormat:
        if self.stimuli_format is None:
            raise InvalidConstructionSequence("add_stimuli")
        return self.stimuli_format

    def to_steps(self, time_in_seconds):
        """Converts a time in seconds to a time in steps"""
        return int(time_in_seconds / self.get_time_step())


class Dataset(_IncompleteDataset):
    """Holds constructed response matrix and stimuli

    [Quick guide to Pandas indexing](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html#min-tut-03-subset)

    Note: You can index straight into a `Dataset` object to get stimuli and
    responses in numpy array format, but you must select rows rather than
    columns. This corresponds to the `df.loc[]` Pandas syntax, rather than
    `df[]`.

    <!--
    >>> import asyncio
    >>> from preconstruct.sources import NeurobankSource
    >>> from preconstruct.stimuliformats import Gammatone
    >>> responses = ['P120_1_1_c92']
    >>> url = 'https://gracula.psyc.virginia.edu/neurobank/'
    >>> stimuli = ['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy',
    ...         'mrel2o09', 'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x']
    >>> data_source = asyncio.run(NeurobankSource.create(url, stimuli, responses))
    >>> builder = DatasetBuilder()
    >>> builder.set_data_source(data_source)
    >>> builder.load_responses()
    >>> builder.bin_responses(time_step=0.005) # 5 ms
    >>> builder.add_stimuli(Gammatone(
    ...     window_time=0.005,
    ...     frequency_bin_count=50,
    ...     min_frequency=500,
    ...     max_frequency=8000,
    ...     log_transform_compress=1,
    ... ))
    >>> builder.create_time_lags(tau=0.3)

    -->
    >>> dataset = builder.get_dataset()
    >>> dataset.responses.index
    Int64Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
               dtype='int64', name='index')
    >>> X, Y = dataset[20:30]
    >>> X.shape
    (4600, 60, 1)
    >>> Y.shape
    (4600, 50)

    You can also manually select directly from the DataFrames.
    """

    responses: pd.DataFrame
    """Columns correspond to pprox files
    """
    stimuli: pd.DataFrame
    """"""
    trial_data: pd.DataFrame
    """Each row corresponds to a trial and each column corresponds to a key in the
    pprox
    """
    time_step: float
    """granularity of time"""
    stimuli_format: StimuliFormat
    """format of the stimuli"""

    def __init__(self, stimuli, responses, trial_data, time_step, stimuli_format):
        self.stimuli = stimuli
        self.responses = responses
        self.trial_data = trial_data
        self.time_step = time_step
        self.stimuli_format = stimuli_format

    def __getitem__(self, key):
        """
        get numpy arrays representing the responses and the stimuli
        at the given pandas index range. The array dimensions are (time, lag, neuron).
        The dimensions of the stimulus will depend on the StimuliFormat chosen
        """
        events = self.get_responses().loc[key]
        responses = np.concatenate(
            [np.stack(x, axis=-1) for x in events.values.tolist()]
        )
        try:
            stimuli_index = self.get_trial_data().loc[key]["stimulus.name"]
        except KeyError:
            stimuli_index = key
        stimuli = self.get_stimuli_format().to_values(
            self.get_stimuli().loc[stimuli_index]
        )
        return responses, stimuli


class DatasetBuilder:
    """Construct instances of the `Dataset` class using the [builder
    pattern](https://refactoring.guru/design-patterns/builder)
    """

    data_source: DataSource
    tau: Optional[float]
    basis: Optional[np.ndarray]

    def __init__(self):
        self._dataset = _IncompleteDataset()
        self.data_source = _EmptySource()

    def set_data_source(self, data_source: DataSource):
        self.data_source = data_source

    def load_responses(self, ignore_columns: Optional[List[str]] = None):
        if ignore_columns is None:
            ignore_columns = []
        clusters = self.data_source.get_responses()
        assert len(clusters) > 0, "no clusters"
        trial_data = pd.concat(
            {
                k: pd.json_normalize(v["pprox"]).set_index("index")
                for k, v in clusters.items()
            },
            axis=1,
        )
        trial_data.columns = trial_data.columns.reorder_levels(order=[1, 0])
        self._dataset.responses = trial_data["events"]
        ignore_columns.append("events")
        trial_data = trial_data.drop(columns=ignore_columns, level=0)
        single_trial = self._aggregate_trials(trial_data)
        assert single_trial is not None
        _, trial_data = single_trial
        self._dataset.trial_data = trial_data

    @staticmethod
    def _aggregate_trials(
        trial_data: pd.DataFrame,
    ) -> Optional[Tuple[str, pd.DataFrame]]:
        """combine corresponding trials across different pprox files

        if all trials contain the same data, return one trial
        otherwise, raise a IncompatibleTrialError
        """
        trials = trial_data.groupby(axis=1, level=1)
        first = None
        for name, t in trials:
            t = t.droplevel(axis=1, level=1)
            if first is None:
                first = (name, t)
            first_name, first_df = first
            if not first_df.equals(t):
                raise IncompatibleTrialError({first_name: first_df, name: t})
        return first

    def bin_responses(self, time_step: float = 0.005):
        """
        transform a point process into bins of size `time_step` containinng
        the number of events that occurred within that time bin.
        """
        self._dataset.time_step = time_step
        self._dataset.responses = self._dataset.get_responses().apply(
            lambda neuron: self._dataset.get_trial_data()
            .join(neuron.rename("events"))
            .apply(self._hist, axis=1)
        )

    def _hist(self, row) -> np.ndarray:
        start, stop = row["interval"]
        time_step = self._dataset.get_time_step()
        # np.arange does not include the end of the interval,
        # so we make the end one time_step later
        bin_edges = np.arange(start, stop + time_step, time_step)
        histogram, _ = np.histogram(row["events"], bin_edges)
        return histogram

    def add_stimuli(self, stimuli_format: StimuliFormat):
        """
        Add a dataframe containing formatted stimuli for each
        stimulus associated with a trial

        Consult documentation for `preconstruct.stimuliformats` for details.

        ###### example
        <!--
        >>> import asyncio
        >>> from preconstruct.sources import NeurobankSource
        >>> responses = ['P120_1_1_c92']
        >>> url = 'https://gracula.psyc.virginia.edu/neurobank/'
        >>> stimuli = ['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy',
        ...         'mrel2o09', 'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x']
        >>> data_source = asyncio.run(NeurobankSource.create(url, stimuli, responses))
        >>> builder = DatasetBuilder()
        >>> builder.set_data_source(data_source)
        >>> builder.load_responses()
        >>> builder.bin_responses(time_step=0.005) # 5 ms

        -->
        >>> from preconstruct.stimuliformats import Gammatone
        >>> builder.add_stimuli(Gammatone(
        ...     window_time=0.001,
        ...     frequency_bin_count=50,
        ...     min_frequency=500,
        ...     max_frequency=8000,
        ... ))
        """
        self._dataset.stimuli_format = stimuli_format
        self._dataset.stimuli = stimuli_format.create_dataframe(
            self.data_source, self._dataset.get_time_step()
        )

    def create_time_lags(self, tau: float = 0.300, basis: Optional[Basis] = None):
        """
        `tau`: length of window (in secs) to consider in prediction
        `basis`: an instance of a class that inherits from
        `preconstruct.basisfunctions.Basis`, initialized with the dimension
        of the projection

        ###### example
        <!--
        >>> import asyncio
        >>> from preconstruct.sources import NeurobankSource
        >>> from preconstruct.stimuliformats import Gammatone
        >>> responses = ['P120_1_1_c92']
        >>> url = 'https://gracula.psyc.virginia.edu/neurobank/'
        >>> stimuli = ['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy',
        ...         'mrel2o09', 'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x']
        >>> data_source = asyncio.run(NeurobankSource.create(url, stimuli, responses))
        >>> builder = DatasetBuilder()
        >>> builder.set_data_source(data_source)
        >>> builder.load_responses()
        >>> builder.bin_responses(time_step=0.005) # 5 ms
        >>> builder.add_stimuli(Gammatone())


        -->
        >>> from preconstruct.basisfunctions import RaisedCosineBasis
        >>> builder.create_time_lags(tau=0.3, basis=RaisedCosineBasis(30))


        """
        self.tau = tau
        self.basis = None
        if basis is not None:
            window_length = self._dataset.to_steps(self.tau)
            self.basis = basis.get_basis(window_length)
        self._dataset.responses = self._dataset.get_responses().apply(
            lambda neuron: self._dataset.get_trial_data()[
                ["stimulus.interval", "stimulus.name"]
            ]
            .join(neuron.rename("events"), on=neuron.index.name)
            .groupby(neuron.index.name)
            .agg("first")
            .reset_index()
            .join(self._dataset.get_stimuli()["stimulus.length"], on="stimulus.name")
            .set_index(neuron.index.name)
            .apply(self._stagger, axis=1)
        )

    def _stagger(self, row):
        stim_start, _ = row["stimulus.interval"]
        start = self._dataset.to_steps(stim_start)
        window_length = self._dataset.to_steps(self.tau)
        stop = start + row["stimulus.length"]
        events = row["events"]
        assert len(events) >= stop - 1 + window_length
        time_lagged = hankel(
            events[start:stop], events[stop - 1 : stop - 1 + window_length]
        )
        if self.basis is not None:
            time_lagged = np.dot(time_lagged, self.basis)
        return time_lagged

    def pool_trials(self):
        """Pool spikes across trials"""
        neurons = self._dataset.get_responses().columns
        if (
            not self._dataset.get_trial_data()
            .groupby("stimulus.name")["stimulus.interval"]
            .apply(lambda ser: len(ser.unique()) == 1)
            .all()
        ):
            raise InconsistentStimulusInterval
        self._dataset.responses = (
            self._dataset.get_responses()
            .join(self._dataset.get_trial_data())
            .groupby("stimulus.name")
            .agg({n: self._trim_to_shortest_and_sum for n in neurons})[neurons]
        )

    @staticmethod
    def _trim_to_shortest_and_sum(ser: pd.Series) -> pd.Series:
        min_length = ser.apply(len).min()
        return ser.apply(lambda x: x[:min_length]).sum()

    def get_dataset(self) -> Dataset:
        """Return the fully constructed `Dataset` object"""
        dataset = self._dataset
        dataset.responses = dataset.get_responses().sort_index()
        return Dataset(
            dataset.get_stimuli(),
            dataset.get_responses(),
            dataset.get_trial_data(),
            dataset.get_time_step(),
            dataset.get_stimuli_format(),
        )


class _EmptySource(DataSource):
    def _get_raw_responses(self):
        self._raise()

    def get_stimuli(self):
        self._raise()

    @staticmethod
    def _raise():
        raise InvalidConstructionSequence("DatasetBuilder.set_data_source")


class InvalidConstructionSequence(Exception):
    """Indicates that the methods of a DatasetBuilder have been called in an invalid order"""

    def __init__(self, required_method):
        super().__init__()
        self.required_method = required_method

    def __str__(self) -> str:
        return (
            f"invalid construction sequence: must call `{self.required_method}` first"
        )


class IncompatibleTrialError(Exception):
    """Raised when corresponding trials from different pprox files differ

    If the differing trial data is expected, you can choose to not include
    specific keys using the `ignore_columns` argument in
    `DatasetBuilder.load_responses`.
    """

    def __init__(self, trial_pair: Dict[str, pd.DataFrame]):
        super().__init__()
        self.trial_pair = trial_pair

    def __str__(self) -> str:
        a, b = tuple(self.trial_pair.keys())
        return (
            f"at least two trials contained conflicting data: {a}, {b}\n"
            f"{self.trial_pair[a].compare(self.trial_pair[b])}"
        )


class InconsistentStimulusInterval(Exception):
    """Raised when stimulus.interval differs between trials that are to be combined"""

    def __str__(self) -> str:
        return (
            "if you want to pool trials, every stimulus presentation must have"
            " the same stimulus.interval"
        )


class IncompatibleStimuliFormat(Exception):
    def __init__(self, format):
        self.format = format

    def __str__(self) -> str:
        return (
            f"the StimuliFormat you have picked ({self.format})"
            " is not a subclass of `SameTimeIndexAsResponse`"
            " so you can't call this function"
        )
