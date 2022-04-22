"""
Stimuli formats
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from gammatone.gtgram import gtgram

from preconstruct.sources import Wav
from preconstruct import _mem

class StimuliFormat(ABC):
    @abstractmethod
    def format_from_wav(self, stimulus: Tuple[str, Wav], time_step: float) -> np.ndarray:
        """return formatted version of WAVE data
        """

    @abstractmethod
    def to_values(self, stim_df) -> np.ndarray:
        """return array representation
        """

    def create_dataframe(self, data_source, time_step) -> pd.DataFrame:
        wav_data = data_source.get_stimuli()
        formatted_stimuli = pd.Series({
            k: self.format_from_wav((k, v), time_step)
            for k, v in wav_data.items()
        })
        stimuli_df = pd.DataFrame()
        stimuli_df["stimulus"] = formatted_stimuli
        return stimuli_df

    def __call__(self, *args) -> np.ndarray:
        return self.format_from_wav(*args)

class SameTimeIndexAsResponse(StimuliFormat):
    """Abstract base class for stimuli formats that use the time index as the responses

    In other words there is a 1 to 1 correspondence between time steps in the stimulus
    and in the response.

    Provides columns `"spectrogram"` and `"stimulus.length"`
    """
    def to_values(self, stim_df) -> np.ndarray:
        return np.concatenate(stim_df["spectrogram"].values)

    def create_dataframe(self, *args) -> pd.DataFrame:
        stimuli_df = super().create_dataframe(*args)
        stimuli_df = stimuli_df.rename(columns={"stimulus": "spectrogram"})
        stimuli_df["stimulus.length"] = \
                stimuli_df["spectrogram"].apply(lambda x: x.shape[0])
        return stimuli_df


    def __call__(self, *args) -> np.ndarray:
        return self.format_from_wav(*args)




class LogTransformable(StimuliFormat):
    def __init__(self, log_transform_compress: Optional[float] = None):
        self.compress = log_transform_compress

    def format_from_wav(self, *args) -> np.ndarray:
        spectrogram = self._raw_format_from_wav(*args)
        if self.compress is not None:
            return np.log10(spectrogram + self.compress) - np.log10(self.compress)
        return spectrogram

    @abstractmethod
    def _raw_format_from_wav(self, stimulus: Tuple[str, Wav], time_step: float) -> np.ndarray:
        """format_from_wav without log_transform applied
        """


class Spectrogram(LogTransformable, SameTimeIndexAsResponse):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def _raw_format_from_wav(self, stimulus: Tuple[str, Wav], time_step: float) -> np.ndarray:
        _name, wav_data = stimulus
        sample_rate, samples = wav_data
        _, _, spectrogram = signal.spectrogram(
                samples,
                sample_rate,
                **self.kwargs
        )
        return spectrogram.T


class Gammatone(LogTransformable, SameTimeIndexAsResponse):
    def __init__(self,
        window_time=0.001,
        frequency_bin_count=50,
        min_frequency=500,
        max_frequency=8000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gammatone_params = {
            "window_time": window_time,
            "channels": frequency_bin_count,
            "f_min": min_frequency,
            "f_max": max_frequency,
        }

    def _raw_format_from_wav(self, stimulus, time_step) -> np.ndarray:
        _name, wav_data = stimulus
        sample_rate, samples = wav_data
        self.gammatone_params["hop_time"] = time_step
        spectrogram = _mem.cache(gtgram)(samples, sample_rate, **self.gammatone_params)
        return spectrogram.T


class Categorical(StimuliFormat):
    def format_from_wav(self, stimulus, time_step) -> np.ndarray:
        name, _wav_data = stimulus
        return name

    def to_values(self, stim_df) -> np.ndarray:
        return pd.get_dummies(stim_df["stimulus"]).values
