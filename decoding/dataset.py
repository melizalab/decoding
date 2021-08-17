"""Loading data for neural decoding
"""
from . import io
import numpy as np
import pandas as pd
from gammatone.gtgram import gtgram
from scipy.linalg import hankel

class DatasetBuilder():
    def __init__(self):
        self._dataset = Dataset()

    def load_responses(self, pprox_path_format, cluster_list=None):
        """
            pprox_path_format: e.g. 'pprox/P120_1_1_{}.pprox'
        """
        clusters = io.load_pprox(pprox_path_format, cluster_list)
        assert len(clusters) > 0, "no clusters"
        activity = pd.concat(
                {k: pd.DataFrame(v['pprox']) \
                        .set_index(['stim','index']) \
                        for k, v in clusters.items()
                },
                axis='columns'
        )
        activity.columns = activity.columns.reorder_levels(order=[1,0])
        self._dataset.activity = activity

    def bin_responses(self, time_step=0.005):
        self._dataset.time_step = time_step
        self._dataset.activity['events'] = self._dataset.activity.groupby(level=1, axis='columns') \
            .apply(lambda x: x.droplevel(1, axis='columns') \
                .apply(self._hist, axis='columns')
            )

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
            wav_path_format: e.g. 'wav/{}.wav'
            window_scale: ratio of gammatone window size to time_step
        """
        self._dataset.window_scale = window_scale
        self._dataset.frequency_bin_count = frequency_bin_count
        self._dataset.min_frequency = min_frequency
        self._dataset.max_frequency = max_frequency
        stimuli_names = set(s for s, _ in self._dataset.activity.index)
        wav_data = io.load_stimuli(wav_path_format, stimuli_names)
        spectrograms = {k: self._spectrogram(v) for k, v in wav_data.items()}
        self._dataset.stimuli = self._dataset.activity.apply(
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
        self._dataset.activity['events'] = self._dataset.activity.groupby(level=1, axis='columns') \
                .apply(lambda x: x.droplevel(1, axis='columns') \
                    .apply(self._stagger, axis='columns')
                )

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

    def pool_trials(self):
        """Pool spikes across trials"""
        events = pd.concat({
                'events': self._dataset.activity['events'].groupby('stim').sum()
            }, axis='columns')
        self._dataset.activity = self._dataset.activity.groupby('stim').first() \
                .drop('events', axis='columns', level=0) \
                .join(events)
        self._dataset.stimuli = self._dataset.stimuli.groupby('stim').first()

    def get_dataset(self):
        dataset = self._dataset
        dataset.activity = dataset.activity.sort_index()
        assert np.array_equal(dataset.activity.index, dataset.stimuli.index)
        dataset.index = dataset.activity.index
        return dataset

class Dataset():
    def __getitem__(self, key):
        """
        get numpy arrays representing the activity and the stimuli
        at the given pandas index range
        """
        events = self.activity.loc[key]['events']
        activity = np.concatenate([np.stack(x,axis=2) for x in events.values.tolist()])
        stimuli = np.concatenate(self.stimuli.loc[key].values)
        return activity, stimuli

    def to_steps(self, time_in_seconds):
        """Converts a time in seconds to a time in steps"""
        return int(time_in_seconds / self.time_step)
