"""Loading data for neural decoding
"""
import os

import numpy as np
import pandas as pd
from joblib import Memory
from appdirs import user_cache_dir
from gammatone.gtgram import gtgram
from scipy.linalg import hankel
from pandarallel import pandarallel

import decoding
from decoding.sources import DataSource

pandarallel.initialize()
_cache_dir = user_cache_dir(decoding.APP_NAME, decoding.APP_AUTHOR)
mem = Memory(_cache_dir, verbose=0)

class DatasetBuilder():
    """Construct instances of the `Dataset` class using the [builder
    pattern](https://refactoring.guru/design-patterns/builder)
    """
    def __init__(self):
        self._dataset = Dataset()
        self.data_source = _EmptySource()
        self.tau = None
        self.basis = None
        self.window_scale = None
        self.max_frequency = None
        self.min_frequency = None

    def set_data_source(self, data_source):
        """ `data_source`: concrete inheritor of `decoding.sources.DataSource`
        """
        self.data_source = data_source

    def load_responses(self):
        clusters = self.data_source.get_responses()
        assert len(clusters) > 0, "no clusters"
        responses = pd.concat(
                {k: pd.DataFrame(v['pprox']) \
                        .set_index(['stim','index']) \
                        for k, v in clusters.items()
                },
                axis='columns'
        )
        responses.columns = responses.columns.reorder_levels(order=[1,0])
        self._dataset.responses = responses

    def bin_responses(self, time_step=0.005):
        """
        transform a time series into bins of size `time_step` containinng
        the number of spikes that occurred within that time bin.
        """
        self._dataset.time_step = time_step
        self._dataset.responses['events'] = self.responses_apply(self._hist)

    def _hist(self, row):
        duration = np.max(row['events']) if len(row['events']) else 0
        # we ignore `duration % time_step` at the end
        bin_edges = np.arange(0, duration, self._dataset.get_time_step())
        histogram, _ = np.histogram(row['events'], bin_edges)
        return histogram

    def add_stimuli(self, window_scale=1, frequency_bin_count=50,
            min_frequency=500, max_frequency=8000, log_transform=True,
            log_transform_compress=1):
        """
            Add a dataframe containing gammatone spectrograms for each
            stimulus associated with a trial

            `window_scale`: ratio of gammatone window size to time_step
            `log_transform`: whether to take the log of the power of each
            spectrogram. If `True`, each point on the spectrogram `x` will
            be transformed into `log(x + log_transform_compress) - log(x)`
        """
        self.window_scale = window_scale
        self.frequency_bin_count = frequency_bin_count
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        wav_data = self.data_source.get_stimuli()
        spectrograms = {k: self._spectrogram(v, log_transform, log_transform_compress) \
                for k, v in wav_data.items()}
        self._dataset.stimuli = self._dataset.get_responses().apply(
                lambda x: spectrograms[x.name[0]],
                axis='columns'
        ).sort_index()

    def _spectrogram(self, wav_data, log_transform, compress):
        sample_rate, samples = wav_data
        spectrogram = mem.cache(gtgram)(
                samples,
                sample_rate,
                window_time=self._dataset.get_time_step()*self.window_scale,
                hop_time=self._dataset.get_time_step(),
                channels=self.frequency_bin_count,
                f_min=self.min_frequency,
                f_max=self.max_frequency
        )
        if log_transform:
            spectrogram = np.log10(spectrogram + compress) - np.log10(compress)
        return spectrogram.T

    def create_time_lags(self, tau=0.300, basis=None):
        """
            `tau`: length of window (in secs) to consider in prediction
            `basis`: an instance of a class that inherits from
            `decoding.basisfunctions.Basis`, initialized with the dimension
            of the projection
        """

        self.tau = tau
        self.basis = basis
        self._dataset.responses['events'] = self.responses_apply(self._stagger)

    def _stagger(self, row):
        start = self._dataset.to_steps(row['stimulus']['interval'][0])
        window_length = self._dataset.to_steps(self.tau)
        stop = start + self._dataset.get_stimuli()[row.name].shape[0]
        total_time_steps = start + stop - 1 + window_length
        pad_width = max(0, total_time_steps - len(row['events']))
        events = np.pad(row['events'], (0, pad_width))
        assert len(events) >= total_time_steps
        time_lagged = hankel(
                events[start:stop],
                events[stop - 1 : stop - 1 + window_length]
        )
        if self.basis is not None:
            basis_matrix = self.basis.get_basis(window_length)
            return np.dot(time_lagged, basis_matrix)
        return time_lagged

    def pool_trials(self):
        """Pool spikes across trials"""
        events = pd.concat({
                'events': self._dataset.get_responses()['events'].groupby('stim').sum()
            }, axis='columns')
        self._dataset.responses = self._dataset.get_responses().groupby('stim').first() \
                .drop('events', axis='columns', level=0) \
                .join(events)
        self._dataset.stimuli = self._dataset.get_stimuli().groupby('stim').first()

    def responses_apply(self, func):
        """
            Responses has a complex structure; this function provides a
            simple way to apply a function to each row

            `func`: (row of responses dataframe) -> (element of output series)

            returns (pandas.Series): the collected outputs of `func`
        """
        return self._dataset.get_responses().groupby(level=1, axis='columns') \
                .parallel_apply(lambda x: x.droplevel(1, axis='columns') \
                    .apply(func, axis='columns')
                )


    def get_dataset(self):
        """Return the fully constructed `Dataset` object"""
        dataset = self._dataset
        dataset.responses = dataset.get_responses().sort_index()
        assert np.array_equal(dataset.get_responses().index, dataset.get_stimuli().index)
        dataset.index = dataset.get_responses().index
        return dataset

class Dataset():
    """Holds constructed response matrix and stimuli

    ### Example usage
    ```
    dataset = dataset_builder.get_dataset()
    X, Y = dataset[:]
    ```
    """
    def __init__(self):
        self.responses = None
        self.stimuli = None
        self.time_step = None

    def get_stimuli(self):
        if self.stimuli is None:
            raise InvalidConstructionSequence("must call `add_stimuli` first")
        return self.stimuli

    def get_time_step(self):
        if self.time_step is None:
            raise InvalidConstructionSequence("must call `bin_responses` first")
        return self.time_step

    def get_responses(self):
        if self.responses is None:
            raise InvalidConstructionSequence("must call `load_responses` first")
        return self.responses

    def __getitem__(self, key):
        """
        get numpy arrays representing the responses and the stimuli
        at the given pandas index range
        """
        events = self.get_responses().loc[key]['events']
        responses = np.concatenate([np.stack(x,axis=2) for x in events.values.tolist()])
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
        raise InvalidConstructionSequence("Must call DatasetBuilder.set_data_source"
                "before using methods that use data")

class InvalidConstructionSequence(Exception):
    """Indicates that the methods of a DatasetBuilder have been called in an invalid order"""
    def __init__(self, description):
        super().__init__()
        self.description = description

    def __str__(self):
        return f"invalid construction sequence: {self.description}"
