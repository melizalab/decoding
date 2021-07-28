"""Loading data for neural decoding
"""
from itertools import chain
from . import io
import numpy as np
from gammatone.gtgram import gtgram

class Dataset():
    def __init__(self, pprox_path_format, wav_path_format,
            time_step=0.005, frequency_bin_count=100,
            min_frequency=200, prediction_window=(5, 15),
            select_clusters=None):
        """
            pprox_path_format: e.g. 'pprox/P120_1_1_{}.pprox'
            wav_path_format: e.g. 'wav/{}.wav'
        """
        self.time_step = time_step
        self.frequency_bin_count = frequency_bin_count
        self.min_frequency = min_frequency
        self.lookahead = prediction_window
        self._build_dataset(
                pprox_path_format,
                wav_path_format,
                select_clusters
        )

    def _get_clusters(self, pprox_path, select_clusters):
        clusters = io.load_pprox(pprox_path, select_clusters)
        assert len(clusters) > 0, "no clusters"
        self.stimulus_from_trial = dict(chain.from_iterable([
            [(trial['index'], trial['stim']) for trial in n['pprox']
        ] for n in clusters.values()]))
        self.stim_on_from_trial = dict(chain.from_iterable([
            [(trial['index'], trial['stim_on']) for trial in n['pprox']
        ] for n in clusters.values()]))
        self.trial_list = self.stimulus_from_trial.keys()
        return clusters

    def _build_dataset(self, pprox_path, wav_path, select_clusters):
        clusters = self._get_clusters(pprox_path, select_clusters)
        stimuli_names = self._get_stimuli_names(clusters)
        stimuli = io.load_stimuli(wav_path, stimuli_names)

        spectrograms = self._get_spectrograms(stimuli)

        binned_spikes_by_trial = self._get_binned_spikes(clusters)

        neural_data, stimuli_data = self._get_data(clusters, binned_spikes_by_trial, spectrograms)
        self.neural_data = neural_data
        self.stimuli_data = stimuli_data

    @staticmethod
    def _get_stimuli_names(clusters):
        """Returns list of all stimuli referred to in pprox data"""
        return np.unique([[trial['stim'] for trial in n['pprox']] for n in clusters.values()])

    def _get_spectrograms(self, stimuli):
        """Returns dictionary of gammatone spectrograms for each stimulus"""
        spectrograms = {}
        for name, (sample_rate, samples) in stimuli.items():
            gammatone_gram = gtgram(
                    samples,
                    sample_rate,
                    window_time=self.time_step,
                    hop_time=self.time_step,
                    channels=self.frequency_bin_count,
                    f_min=self.min_frequency
            )
            spectrograms[name] = gammatone_gram
        return spectrograms

    def _get_binned_spikes(self, clusters):
        n_neurons = len(clusters)
        binned_spikes_by_trial = {}
        for n, neuron in enumerate(clusters.values()):
            sampling_rate = neuron['entry_metadata'][0]['sampling_rate']
            for trial in neuron['pprox']:
                bin_count = self._get_bin_count(trial, sampling_rate)
                spikes, _ = np.histogram( trial['events'], bins=bin_count)

                binned = binned_spikes_by_trial.get(trial['index'], np.zeros((bin_count, n_neurons)))
                binned[:, n] = spikes
                binned_spikes_by_trial[trial['index']] = binned
        return binned_spikes_by_trial

    def _get_bin_count(self, trial, sampling_rate):
        spike_duration = (trial['recording']['stop'] - trial['recording']['start']) \
                / sampling_rate
        return int(spike_duration / self.time_step)

    def _get_stim_on(self, trial):
        return int(self.stim_on_from_trial[trial] / self.time_step)

    def _get_data(self, clusters, binned_spikes_by_trial, spectrograms):
        neural_data = []
        stimuli_data = []
        for trial in self.trial_list:
            stimulus = spectrograms[self.stimulus_from_trial[trial]].T
            stim_on = self._get_stim_on(trial)
            for step, stim in enumerate(stimulus, start=stim_on):
                start, stop = self.lookahead
                window_start = step + start
                window_stop = step + stop
                neural_data.append(binned_spikes_by_trial[trial][window_start:window_stop])
                stimuli_data.append(stim)
        return np.array(neural_data), np.array(stimuli_data)
