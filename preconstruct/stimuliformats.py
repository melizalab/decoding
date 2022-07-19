"""
Classes that allow you to configure different ways of representing stimuli

Only concrete classes (not abstract) can be used as input to a
`preconstruct.dataset.DatasetBuilder`
"""
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Mapping, Callable

import numpy as np
import pandas as pd
from scipy import signal
from gammatone.gtgram import gtgram
from gammatone.filters import centre_freqs

from preconstruct.sources import Wav, DataSource
from preconstruct import _mem


class StimuliFormat(ABC):
    """Abstract base class for representing stimuli"""

    @abstractmethod
    def format_from_wav(
        self, name: str, wav_data: Wav, interval: Tuple[float, float], time_step: float
    ) -> pd.DataFrame:
        """return formatted version of WAVE data"""

    def to_values(self, stim_df) -> np.ndarray:
        """convert from pd.DataFrame to ndarray"""
        return stim_df.values

    def create_dataframe(
        self,
        data_source: DataSource,
        time_step: float,
        intervals: pd.Series,
    ) -> pd.DataFrame:
        """build `stimuli` DataFrame"""
        wav_data = data_source.get_stimuli()
        stimuli = (
            pd.DataFrame(wav_data, index=["sample_rate", "samples"])
            .T.rename_axis("stimulus.name")
            .join(intervals.rename("interval"), how="right")
        )
        formatted_stimuli = pd.concat(
            {
                index: self.format_from_wav(
                    index,
                    (data["sample_rate"], data["samples"]),
                    data["interval"],
                    time_step,
                )
                for index, data in stimuli.iterrows()
            }
        ).fillna(0)
        return formatted_stimuli


class LogTransformable(StimuliFormat):
    """Abstract base class for representing stimuli that can be log transformed

    provides the log_transform_compress keyword argument
    """

    def __init__(self, log_transform_compress: Optional[float] = None, **_kwargs):
        self.compress = log_transform_compress

    def format_from_wav(self, *args) -> pd.DataFrame:
        spectrogram = self._raw_format_from_wav(*args)
        if self.compress is not None:
            return np.log10(spectrogram + self.compress) - np.log10(self.compress)
        return spectrogram

    @abstractmethod
    def _raw_format_from_wav(
        self, wav_data: Wav, interval: Tuple[float, float], time_step: float
    ) -> pd.DataFrame:
        """format_from_wav without log_transform applied"""


class Spectrogram(LogTransformable):
    """Spectrogram format

    Consult the [scipy documentation for spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html)
    to see a list of acceptable keywords. (`x`, `fs`, and `nperseg` will be set automatically.)
    """

    def __init__(
        self,
        min_frequency: Optional[int] = None,
        max_frequency: Optional[int] = None,
        log_transform_compress: Optional[float] = None,
        **kwargs
    ):
        self.kwargs = kwargs
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        super().__init__(log_transform_compress, **kwargs)

    def _raw_format_from_wav(
        self, _name: str, wav_data: Wav, interval: Tuple[float, float], time_step: float
    ) -> pd.DataFrame:
        sample_rate, samples = wav_data
        nperseg = int(sample_rate * time_step)
        f, t, spectrogram = signal.spectrogram(
            samples, sample_rate, nperseg=nperseg, **self.kwargs
        )
        start_time, _ = interval
        spectrogram_df = pd.DataFrame(spectrogram.T, columns=f, index=t + start_time)
        if self.min_frequency:
            spectrogram_df = spectrogram_df[
                spectrogram_df.columns[spectrogram_df.columns >= self.min_frequency]
            ]
        if self.max_frequency:
            spectrogram_df = spectrogram_df[
                spectrogram_df.columns[spectrogram_df.columns <= self.max_frequency]
            ]
        return spectrogram_df


class Gammatone(LogTransformable):
    """Gammatone format"""

    def __init__(
        self,
        window_time=0.001,
        frequency_bin_count=50,
        min_frequency=500,
        max_frequency=8000,
        log_transform_compress: Optional[float] = None,
    ):
        super().__init__(log_transform_compress)
        self.params = {
            "window_time": window_time,
            "channels": frequency_bin_count,
            "f_min": min_frequency,
            "f_max": max_frequency,
        }

    def _raw_format_from_wav(
        self, _name, wav_data: Wav, interval: Tuple[float, float], time_step: float
    ) -> pd.DataFrame:
        sample_rate, samples = wav_data
        self.params["hop_time"] = time_step
        spectrogram = _mem.cache(gtgram)(samples, sample_rate, **self.params)
        return pd.DataFrame(
            spectrogram.T,
            columns=centre_freqs(
                sample_rate,
                self.params["channels"],
                self.params["f_min"],
                self.params["f_max"],
            ),
            index=np.linspace(*interval, spectrogram.shape[1]),
        )


class SyllableCategorical(StimuliFormat):
    """Each syllable gets its own identifier"""

    def __init__(self, **kwargs) -> None:
        """kwargs passed to scipy.signal.find_peaks"""
        super().__init__()
        self.kwargs = kwargs

    def format_from_wav(
        self, name: str, wav_data: Wav, interval: Tuple[float, float], time_step: float
    ) -> pd.DataFrame:
        """return formatted version of WAVE data"""
        start, stop = interval
        peaks = self.find_syllable_boundaries(wav_data, interval, time_step)
        syllable_boundaries = pd.Series(
            [
                start,
                *peaks,
                stop + time_step,
            ]
        )
        t = np.arange(start, stop + time_step, time_step)
        syllables = pd.get_dummies(
            pd.cut(pd.Series(t, index=t), bins=syllable_boundaries, include_lowest=True)
        ).rename(columns=lambda x: (name, x))
        return syllables

    def find_syllable_boundaries(self, wav_data, interval, time_step):
        sample_rate, samples = wav_data
        start, stop = interval
        # we want smaller windows than the size of time_step for identifying syllables
        # with greater precision
        nperseg = int(sample_rate * time_step) // 2
        _, _, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg)
        power = -np.mean(np.log10(spectrogram + 1), axis=0)
        # syllable boundaries should be moments when the sound gets quiet for a bit
        find_peaks_kwargs = {
            "height": np.median(power),
            "distance": 5,
            "prominence": np.std(power),
        }
        find_peaks_kwargs.update(self.kwargs)
        peaks, _ = signal.find_peaks(power, **find_peaks_kwargs)
        return np.linspace(start, stop, len(power))[peaks]
