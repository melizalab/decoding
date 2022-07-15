"""
Classes that allow you to configure different ways of representing stimuli

Only concrete classes (not abstract) can be used as input to a
`preconstruct.dataset.DatasetBuilder`
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal
from gammatone.gtgram import gtgram

from preconstruct.sources import Wav
from preconstruct import _mem


class StimuliFormat(ABC):
    """Abstract base class for representing stimuli"""

    @abstractmethod
    def format_from_wav(self, name: str, wav_data: Wav, time_step: float) -> np.ndarray:
        """return formatted version of WAVE data"""

    def to_values(self, stim_df) -> np.ndarray:
        """convert from pd.DataFrame to ndarray"""
        return np.concatenate(stim_df["stimulus"].values)

    def create_dataframe(self, data_source, time_step) -> pd.DataFrame:
        """build `stimuli` DataFrame"""
        wav_data = data_source.get_stimuli()
        stimuli_df = pd.DataFrame()
        stimuli_df["stimulus"] = pd.Series(
            {k: self.format_from_wav(k, v, time_step) for k, v in wav_data.items()}
        )
        stimuli_df["stimulus.length"] = stimuli_df["stimulus"].apply(
            lambda x: x.shape[0]
        )
        return stimuli_df


class LogTransformable(StimuliFormat):
    """Abstract base class for representing stimuli that can be log transformed

    provides the log_transform_compress keyword argument
    """

    def __init__(self, log_transform_compress: Optional[float] = None, **_kwargs):
        self.compress = log_transform_compress

    def format_from_wav(self, *args) -> np.ndarray:
        spectrogram = self._raw_format_from_wav(*args)
        if self.compress is not None:
            return np.log10(spectrogram + self.compress) - np.log10(self.compress)
        return spectrogram

    @abstractmethod
    def _raw_format_from_wav(
        self, name: str, wav_data: Wav, time_step: float
    ) -> np.ndarray:
        """format_from_wav without log_transform applied"""


class Spectrogram(LogTransformable):
    """Spectrogram format

    Consult the [scipy documentation for spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html)
    to see a list of acceptable keywords. (`x`, `fs`, and `nperseg` will be set automatically.)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def _raw_format_from_wav(
        self, _name: str, wav_data: Wav, time_step: float
    ) -> np.ndarray:
        sample_rate, samples = wav_data
        nperseg = int(sample_rate * time_step)
        _, _, spectrogram = signal.spectrogram(
            samples, sample_rate, nperseg=nperseg, **self.kwargs
        )
        return spectrogram.T


class Gammatone(LogTransformable):
    """Gammatone format"""

    def __init__(
        self,
        window_time=0.001,
        frequency_bin_count=50,
        min_frequency=500,
        max_frequency=8000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.params = {
            "window_time": window_time,
            "channels": frequency_bin_count,
            "f_min": min_frequency,
            "f_max": max_frequency,
        }

    def _raw_format_from_wav(self, _name, wav_data, time_step) -> np.ndarray:
        sample_rate, samples = wav_data
        self.params["hop_time"] = time_step
        spectrogram = _mem.cache(gtgram)(samples, sample_rate, **self.params)
        return spectrogram.T
