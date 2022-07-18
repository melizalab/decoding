from preconstruct import DatasetBuilder
from preconstruct.sources import MemorySource
from preconstruct.stimuliformats import *


def test_spectrogram(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.bin_responses()
    min_frequency = 1000
    max_frequency = 8000
    builder.add_stimuli(
        Spectrogram(
            scaling="density",
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            log_transform_compress=1,
        )
    )
    builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert X.shape[0] == Y.shape[0]
    frequency_bands = dataset.get_stimuli().columns
    assert ((frequency_bands > min_frequency) & (frequency_bands < max_frequency)).all()


def test_gammatone(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.bin_responses()
    frequency_bin_count = 50
    builder.add_stimuli(Gammatone(frequency_bin_count=frequency_bin_count))
    builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert Y.shape[1] == frequency_bin_count
    assert X.shape[0] == Y.shape[0]


def test_syllable(real_data_source):
    builder = DatasetBuilder()
    builder.set_data_source(real_data_source)
    builder.load_responses()
    builder.bin_responses()

    def peak_finder(wav_data, interval, time_step):
        sample_rate, samples = wav_data
        start, stop = interval
        # we want smaller windows than the size of time_step for identifying syllables
        # with greater precision
        nperseg = int(sample_rate * time_step) // 2
        _, _, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg)
        power = np.mean(spectrogram, axis=0)
        # syllable boundaries should be moments when the sound gets quiet for a bit
        delta_power = power[1:] - power[:-1]
        peaks, _ = signal.find_peaks(delta_power, height=np.mean(np.abs(delta_power)))
        return np.linspace(start, stop, delta_power.shape[0])[peaks]

    builder.add_stimuli(SyllableCategorical(peak_finder))
    builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert X.shape[0] == Y.shape[0]


def test_datasource_has_diff_stimuli(stimtrial_pprox):
    sample_rate = 44100
    samples = np.random.normal(0, 1000, 44100)
    # note that the stimuli dict contains a strict superset of the stimuli referenced
    # in the responses dict. That's the point of this test
    stimuli = {"song_1": (sample_rate, samples), "song_2": (sample_rate, samples)}
    responses = stimtrial_pprox
    builder = DatasetBuilder()
    builder.set_data_source(MemorySource(responses, stimuli))
    builder.load_responses()
    builder.bin_responses()
    frequency_bin_count = 50
    builder.add_stimuli(Gammatone(frequency_bin_count=frequency_bin_count))
    builder.create_time_lags()
    dataset = builder.get_dataset()
    X, Y = dataset[:]
    assert Y.shape[1] == frequency_bin_count
    assert X.shape[0] == Y.shape[0]
