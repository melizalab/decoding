"""Loading data for neural decoding
"""
from itertools import chain
from . import io
import numpy as np
import pandas as pd
from gammatone.gtgram import gtgram
from scipy.linalg import hankel

class Dataset():
    def __init__(self, pprox_path_format, wav_path_format,
            time_step=0.005, frequency_bin_count=100,
            min_frequency=200, max_frequency=8000,
            tau=0.050, window_scale=1):
        """
            pprox_path_format: e.g. 'pprox/P120_1_1_{}.pprox'
            wav_path_format: e.g. 'wav/{}.wav'
            tau: length of window (in secs) to consider in prediction
            window_scale: ratio of gammatone window size to time_step
        """
        self.time_step = time_step
        self.window_scale = window_scale
        self.frequency_bin_count = frequency_bin_count
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.tau = tau

        clusters = io.load_pprox(pprox_path_format)
        assert len(clusters) > 0, "no clusters"
        # for simplicity, assume every pprox has same sampling rate
        arbitrary_cluster = next(iter(clusters.values()))
        self.neural_sampling_rate = arbitrary_cluster['entry_metadata'][0]['sampling_rate']
        activity = pd.concat(
                {k: pd.DataFrame(v['pprox']) \
                        .set_index(['stim','index']) \
                        for k, v in clusters.items()
                },
                axis='columns'
        )
        activity.columns = activity.columns.reorder_levels(order=[1,0])
        activity['events'] = activity.groupby(level=1, axis='columns') \
            .apply(lambda x: x.droplevel(1, axis='columns') \
                .apply(self._hist, axis='columns')
            )
        stimuli_names = set(s for s, _ in activity.index)
        wav_data = pd.Series(io.load_stimuli(wav_path_format, stimuli_names))
        self.stimuli = activity.apply(
                lambda x: self._spectrogram(x.name[0], wav_data),
                axis='columns'
        ).sort_index()
        activity['events'] = activity.groupby(level=1, axis='columns') \
                .apply(lambda x: x.droplevel(1, axis='columns') \
                    .apply(self._stagger, axis='columns')
                )
        self.activity = activity.sort_index()
        assert np.array_equal(self.activity.index, self.stimuli.index)


    def __getitem__(self, key):
        """
        get numpy arrays representing the activity and the stimuli
        at the given pandas index range
        """
        events = self.activity.loc[key]['events']
        activity = np.concatenate([np.stack(x,axis=2) for x in events.values.tolist()])
        stimuli = np.concatenate(self.stimuli.loc[key].values)
        return activity, stimuli

    def _spectrogram(self, stimulus, wav_data):
        sample_rate, samples = wav_data[stimulus]
        spectrogram = gtgram(
                samples,
                sample_rate,
                window_time=self.time_step*self.window_scale,
                hop_time=self.time_step,
                channels=self.frequency_bin_count,
                f_min=self.min_frequency,
                f_max=self.max_frequency
        )
        return spectrogram.T

    def _hist(self, row):
        duration = (row['recording']['stop'] - row['recording']['start']) \
                / self.neural_sampling_rate
        # we ignore `duration % time_step` at the end
        bin_edges = np.arange(0, duration, self.time_step)
        histogram, _ = np.histogram(row['events'], bin_edges)
        return histogram

    def _stagger(self, row):
        start = self.to_steps(row['stim_on'])
        window_length = self.to_steps(self.tau)
        stop = start + self.stimuli[row.name].shape[0]
        return hankel(
                row['events'][start:stop],
                row['events'][stop - 1 : stop - 1 + window_length]
        )

    def to_steps(self, time_in_seconds):
        """Converts a time in seconds to a time in steps"""
        return int(time_in_seconds / self.time_step)
