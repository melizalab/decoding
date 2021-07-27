"""Loading data for neural decoding
"""
from itertools import chain
from . import io
import numpy as np
from gammatone.gtgram import gtgram

class Dataset():
    def __init__(self, time_step=0.005):
        self.time_step = time_step
        self.frequency_bin_count = 100
        self.min_frequency = 200
        self.lookahead = (5, 15)
        self._build_dataset()

    def _build_dataset(self):
        clusters = io.load_pprox('../pprox/P120_1_1_{}.pprox')
        auditory_neurons = set(['c244','c314','c329','c362','c89','c92'])
        clusters = dict(filter(lambda x: x[0] in auditory_neurons, clusters.items()))

        assert len(clusters) > 0, "no clusters"

        stimuli_names = self.get_stimuli_names(clusters)
        stimuli = io.load_stimuli('../wav/{}.wav', stimuli_names)

        spectrograms = self.get_spectrograms(stimuli)

        binned_spikes_by_trial = self.get_binned_spikes(clusters)

        neural_data, stimuli_data = self.get_data(clusters, binned_spikes_by_trial, spectrograms)
        self.neural_data = neural_data
        self.stimuli_data = stimuli_data

    @staticmethod
    def get_stimuli_names(clusters):
        """Returns list of all stimuli referred to in pprox data"""
        return np.unique([[trial['stim'] for trial in n['pprox']] for n in clusters.values()])

    def get_spectrograms(self, stimuli):
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

    def get_binned_spikes(self, clusters):
        n_neurons = len(clusters)
        binned_spikes_by_trial = {}
        for n, neuron in enumerate(clusters.values()):
            sampling_rate = neuron['entry_metadata'][0]['sampling_rate']
            for trial in neuron['pprox']:
                spike_duration = (trial['recording']['stop'] - trial['recording']['start']) \
                        / sampling_rate
                bin_count = int(spike_duration / self.time_step)
                spikes, _ = np.histogram(trial['events'], bins=bin_count)

                bt = binned_spikes_by_trial.get(trial['index'], np.zeros((bin_count, n_neurons)))
                bt[:, n] = spikes
                binned_spikes_by_trial[trial['index']] = bt
        return binned_spikes_by_trial

    def get_data(self, clusters, binned_spikes_by_trial, spectrograms):
        stimulus_from_trial = dict(chain.from_iterable([
            [(trial['index'], trial['stim']) for trial in n['pprox']
        ] for n in clusters.values()]))
        stim_on_from_trial = dict(chain.from_iterable([
            [(trial['index'], trial['stim_on']) for trial in n['pprox']
        ] for n in clusters.values()]))
        trial_list = stimulus_from_trial.keys()
        assert len(trial_list) == 100
        neural_data = []
        stimuli_data = []
        for trial in trial_list:
            stimulus = spectrograms[stimulus_from_trial[trial]].T
            stim_on = int(stim_on_from_trial[trial] / self.time_step)
            for step, stim in enumerate(stimulus, start=stim_on):
                start, stop = self.lookahead
                window_start = step + start
                window_stop = step + stop
                neural_data.append(binned_spikes_by_trial[trial][window_start:window_stop])
                stimuli_data.append(stim)
        return np.array(neural_data), np.array(stimuli_data)
