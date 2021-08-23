"""Loading data for neural decoding
"""
from . import io
import numpy as np
import pandas as pd
from gammatone.gtgram import gtgram
from scipy.linalg import hankel

class DatasetBuilder():
    """Construct instances of the `Dataset` class using the builder
    pattern (https://refactoring.guru/design-patterns/builder)
    """
    def __init__(self):
        self._dataset = Dataset()

    def load_responses(self, pprox_path_format, cluster_list=None):
        """
            load pprox data

            pprox_path_format: e.g. 'pprox/P120_1_1_{}.pprox'
            cluster_list: optional list of identifiers (that fill in
            the {} in `pprox_path_format`) to load from the path format.
            If not provided, load all matching files.
        """
        clusters = io.load_pprox(pprox_path_format, cluster_list)
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
        bin_edges = np.arange(0, duration, self._dataset.time_step)
        histogram, _ = np.histogram(row['events'], bin_edges)
        return histogram

    def add_stimuli(self, wav_path_format, stimuli_names=None,
            window_scale=1, frequency_bin_count=50, min_frequency=200,
            max_frequency=8000):
        """
            Add a dataframe containing gammatone spectrograms for each
            stimulus associated with a trial

            wav_path_format: e.g. 'wav/{}.wav'
            window_scale: ratio of gammatone window size to time_step
            stimuli_names: optional list of stimuli identifiers to load.
            Loads all matching files if not provided.
        """
        self._dataset.window_scale = window_scale
        self._dataset.frequency_bin_count = frequency_bin_count
        self._dataset.min_frequency = min_frequency
        self._dataset.max_frequency = max_frequency
        stimuli_names = set(s for s, _ in self._dataset.responses.index)
        wav_data = io.load_stimuli(wav_path_format, stimuli_names)
        spectrograms = {k: self._spectrogram(v) for k, v in wav_data.items()}
        self._dataset.stimuli = self._dataset.responses.apply(
                lambda x: spectrograms[x.name[0]],
                axis='columns'
        ).sort_index()

    def _spectrogram(self, wav_data):
        sample_rate, samples = wav_data
        spectrogram = gtgram(
                samples,
                sample_rate,
                window_time=self._dataset.time_step*self._dataset.window_scale,
                hop_time=self._dataset.time_step,
                channels=self._dataset.frequency_bin_count,
                f_min=self._dataset.min_frequency,
                f_max=self._dataset.max_frequency
        )
        return spectrogram.T

    def create_time_lags(self, tau=0.300):
        """
            tau: length of window (in secs) to consider in prediction
        """

        self._dataset.tau = tau
        self._dataset.responses['events'] = self.responses_apply(self._stagger)

    def _stagger(self, row):
        start = self._dataset.to_steps(row['stim_on'])
        window_length = self._dataset.to_steps(self._dataset.tau)
        stop = start + self._dataset.stimuli[row.name].shape[0]
        total_time_steps = start + stop - 1 + window_length
        pad_width = max(0, total_time_steps - len(row['events']))
        events = np.pad(row['events'], (0, pad_width))
        assert len(events) >= total_time_steps
        return hankel(
                events[start:stop],
                events[stop - 1 : stop - 1 + window_length]
        )

    def project_responses(self, basis, num_basis_functions, **kwargs):
        """
        optionally project binned responses to a new basis

        basis: a class that inherits from decoding.basisfunctions.Basis
        """
        num_timesteps = self._dataset.to_steps(self._dataset.tau)
        _basis = basis(num_basis_functions, num_timesteps, **kwargs)
        project = lambda row: np.dot(row['events'], _basis.get_basis())
        self._dataset.responses['events'] = self.responses_apply(project)

    def pool_trials(self):
        """Pool spikes across trials"""
        events = pd.concat({
                'events': self._dataset.responses['events'].groupby('stim').sum()
            }, axis='columns')
        self._dataset.responses = self._dataset.responses.groupby('stim').first() \
                .drop('events', axis='columns', level=0) \
                .join(events)
        self._dataset.stimuli = self._dataset.stimuli.groupby('stim').first()

    def responses_apply(self, func):
        """
            Responses has a complex structure; this function provides a
            simple way to apply a function to each row

            function: (row of responses dataframe) -> (element of output series)

            returns (pandas.Series): the collected outputs of `func`
        """
        return self._dataset.responses.groupby(level=1, axis='columns') \
                .apply(lambda x: x.droplevel(1, axis='columns') \
                    .apply(func, axis='columns')
                )


    def get_dataset(self):
        """Return the fully constructed `Dataset` object"""
        dataset = self._dataset
        dataset.responses = dataset.responses.sort_index()
        assert np.array_equal(dataset.responses.index, dataset.stimuli.index)
        dataset.index = dataset.responses.index
        return dataset

class Dataset():
    def __getitem__(self, key):
        """
        get numpy arrays representing the responses and the stimuli
        at the given pandas index range
        """
        events = self.responses.loc[key]['events']
        responses = np.concatenate([np.stack(x,axis=2) for x in events.values.tolist()])
        stimuli = np.concatenate(self.stimuli.loc[key].values)
        return responses, stimuli

    def to_steps(self, time_in_seconds):
        """Converts a time in seconds to a time in steps"""
        return int(time_in_seconds / self.time_step)
