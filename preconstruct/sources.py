"""Data sources

## Examples

Let's build a simple application that accepts input in the form of a DataSource.

>>> import numpy as np
>>> from preconstruct.sources import DataSource, MemorySource
>>> def stim_resp_summary(data_source: DataSource):
...     stimuli = data_source.get_stimuli()
...     print('stimuli sample rates:')
...     for name, (fs, _samples) in stimuli.items():
...         print(f'{name}: {fs}')
...     responses = data_source.get_responses()
...     print('responses trial count:')
...     for name, pprox in responses.items():
...         print(f'{name}: {len(pprox["pprox"])}')

>>> responses = ['P120_1_1_c92']
>>> url = 'https://gracula.psyc.virginia.edu/neurobank/'
>>> stimuli = ['c95zqjxq', 'g29wxi4q', 'igmi8fxa', 'jkexyrd5', 'l1a3ltpy',
...         'mrel2o09', 'p1mrfhop', 'vekibwgj', 'w08e1crn', 'ztqee46x']
>>> data_source = asyncio.run(NeurobankSource.create(url, stimuli, responses))
>>> stim_resp_summary(data_source)
stimuli sample rates:
c95zqjxq: 44100
g29wxi4q: 44100
igmi8fxa: 44100
jkexyrd5: 44100
l1a3ltpy: 44100
mrel2o09: 44100
p1mrfhop: 44100
vekibwgj: 44100
w08e1crn: 44100
ztqee46x: 44100
responses trial count:
P120_1_1_c92: 100
"""
import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Generator, Union, \
        TypedDict, Set
from abc import ABC, abstractmethod
from pathlib import Path
from glob import glob
import json
from urllib.parse import urljoin, urlparse

import numpy as np
import aiohttp
import parse
from scipy.io import wavfile
from appdirs import user_cache_dir


from preconstruct import APP_NAME, APP_AUTHOR
from . import dataset

class StimConfig(TypedDict):
    """type annotation for information about a stimulus presentation"""
    name: str
    interval: Tuple[float]

class Trial(TypedDict):
    """type annotation for a trial"""
    events: List[float]
    interval: Tuple[float]
    stimulus: StimConfig
    offset: Optional[float]
    index: Optional[int]

class Pprox(TypedDict):
    """type annotation for a collection of trials"""
    pprox: List[Trial]

Wav = Tuple[int, np.ndarray]

class DataSource(ABC):
    """Abstract class for data sources to inherit from

    The methods on this class will be available from all the child classes
    in the module, but you cannot directly instantiate this class.

    Concrete child implementations must define how to load data (e.g. local path, URL).
    By inheriting from this class, they provide a standardized interface to
    access the data
    """

    @abstractmethod
    def _get_raw_responses(self) -> Dict[str, Pprox]:
        """returns dictionary mapping names of files to pprox data

        pprox may be in the incorrect format
        """

    def get_responses(self) -> Dict[str, Pprox]:
        """returns dictionary mapping names of files to stimtrial pprox data"""
        responses = self._get_raw_responses()
        stimuli = self.get_stimuli()
        durations = {name: len(s) / fs for name, (fs, s) in stimuli.items()}
        return _fix_pprox(responses, durations)

    @abstractmethod
    def get_stimuli(self) -> Dict[str, Wav]:
        """returns dictionary mapping names of files to (sample_rate, samples) tuples"""

    def __eq__(self, other) -> bool:
        return (self.get_responses() == other.get_responses()) and (
            self.get_stimuli() == other.get_stimuli()
        )

    def stimuli_names_from_pprox(self) -> Set[str]:
        """returns a set of all stimuli used in the responses"""
        return set(self._stimuli_generator())


    def _stimuli_generator(self) -> Generator[str, None, None]:
        durations = defaultdict(lambda: 0.0)
        responses = _fix_pprox(self._get_raw_responses(), durations)
        for pprox in responses.values():
            for trial in pprox['pprox']:
                yield trial['stimulus']['name']


class FsSource(DataSource):
    """Loads data from local File System (FS) """

    def __init__(
        self,
        pprox_path_format: Union[str, Path],
        wav_path_format: Union[str, Path],
        stimuli_names: Optional[List[str]] = None,
        cluster_list: Optional[List[str]] = None,
    ):
        """initialize with path to data

        Paths are specified with a format string, where `{}` is used to
        represent the part of the path/filename that determines the identity of
        the file contents. This way, files identifiers do not need to contain
        redundant information, such as file extension. If a list of identifers
        is not specified, the wildcard `*` will be substituted for `{}` to load
        all matching files.

        `pprox_path_format`: e.g. `'pprox/P120_1_1_{}.pprox'`

        `wav_path_format`: e.g. `'wav/{}.wav'`

        `cluster_list`: optional list of response identifiers to load

        `stimuli_names`: optional list of stimuli identifiers to load
        """
        if not isinstance(pprox_path_format, str):
            pprox_path_format = str(pprox_path_format)
        if not isinstance(wav_path_format, str):
            wav_path_format = str(wav_path_format)
        self.pprox_path_format = pprox_path_format
        self.cluster_list = cluster_list
        self.wav_path_format = wav_path_format
        self.stimuli_names = stimuli_names

    def _get_raw_responses(self):
        return self._load_pprox(
            self.pprox_path_format,
            cluster_names=self.cluster_list,
        )

    def get_stimuli(self) -> Dict[str, Wav]:
        return self._load_stimuli(self.wav_path_format, self.stimuli_names)

    @staticmethod
    def _load_stimuli(path_format, stimuli_names=None):
        """Load wav files into a dictionary

        stimuli_names: iterable of strings
        path_format: path containing "{}" which will be replaced by
                     each of `stimuli_names`

        returns: dictionary mapping stimuli_names to (sample_rate, samples)
        """
        stimuli = {}
        for name, filename in FsSource._get_filenames(path_format, stimuli_names):
            sample_rate, samples = wavfile.read(filename)
            stimuli[name] = (sample_rate, samples)
        return stimuli

    @staticmethod
    def _load_pprox(path_format, cluster_names=None):
        """Load pprox files into a dictionary
        path_format: path containing "{}" which will be globbed
                     for all matching files and then parsed as
                     JSON
        cluster_names (iterable): if specified, load only the clusters
                       with the given names
        returns: dictionary mapping the part of the filenames
                 represented by "{}" to their parsed contents
        """
        clusters = {}
        for name, path in FsSource._get_filenames(path_format, cluster_names):
            with open(path, "r") as pprox_file:
                try:
                    json_data = json.load(pprox_file)
                except UnicodeDecodeError as exc:
                    raise ValueError(
                            "could not load pprox files as text data"
                            " (are you putting stimuli where responses should go?)"
                    ) from exc
                clusters[name] = json_data
        return clusters

    @staticmethod
    def _get_filenames(path_format, names):
        assert path_format.find("{}") != -1, (
            "path_format should include empty braces where a wildcard"
            " would go.\n"
            'For example, "{}.pprox" will load all files in the current'
            ' directory that end in ".pprox". The contents of each file'
            ' will be named with the part of the filename before ".pprox".'
        )
        parser = parse.compile(path_format)
        if names is None:
            filenames = glob(path_format.format("*"))
            names = map(lambda p: parser.parse(p)[0], filenames)
        else:
            filenames = map(path_format.format, names)
        return zip(names, filenames)


class NeurobankSource(FsSource):
    """Downloads data from Neurobank and caches the files

    `NeurobankSource` uses asynchronous network calls in order to download
    resources in parallel. As a result, it must be initialized in a special
    way:
    ```
    >>> import asyncio
    >>> from preconstruct.sources import NeurobankSource
    >>> stimuli = ['ztqee46x', '00oagdl5', 'g29wxi4q', 'mrel2o09', 'vekibwgj', \
        'l1a3ltpy', 'igmi8fxa', 'c95zqjxq', 'w08e1crn', 'jkexyrd5', 'p1mrfhop', ]
    >>> responses = ['P120_1_1_c92']
    >>> url = 'https://gracula.psyc.virginia.edu/neurobank/'
    >>> data_source = asyncio.run(NeurobankSource.create(url, stimuli, responses))

    ```

    In a Jupyter Notebook, you can skip importing `asyncio` and just run:
    ```
    data_source = await NeurobankSource.create(url, stimuli, responses)
    ```
    """

    _DOWNLOAD_FORMAT = "resources/{}/download"

    @classmethod
    async def create(cls, neurobank_registry: str, wav_ids, pprox_ids):
        """
        `neurobank_registry`: URL of Neurobank instance

        both `pprox_ids` and `wav_ids` may either be a
        list of resource IDs, or path to a file containing such a list
        """
        url_format = urljoin(neurobank_registry, cls._DOWNLOAD_FORMAT)
        pprox_ids = cls._get_list(pprox_ids)
        wav_ids = cls._get_list(wav_ids)
        parsed_url = urlparse(neurobank_registry)
        cache_dir = Path(user_cache_dir(APP_NAME, APP_AUTHOR)) / parsed_url.netloc
        self = NeurobankSource(url_format, pprox_ids, wav_ids, cache_dir)
        responses_dir = self.cache_dir / "responses"
        stimuli_dir = self.cache_dir / "stimuli"
        responses_dir.mkdir(parents=True, exist_ok=True)
        stimuli_dir.mkdir(parents=True, exist_ok=True)
        await self._download_all()
        super().__init__(
            self,
            responses_dir / "{}",
            stimuli_dir / "{}",
            cluster_list=self.pprox_ids,
            stimuli_names=self.wav_ids,
        )
        return self

    def __init__(self, url_format, pprox_ids, wav_ids, cache_dir):
        """`NeurobankSource` cannot be directly initialized.
        Use `NeurobankSource.create` instead."""
        self.url_format = url_format
        self.pprox_ids = pprox_ids
        self.wav_ids = wav_ids
        self.cache_dir = cache_dir

    @staticmethod
    def _get_list(resource_ids):
        if isinstance(resource_ids, list):
            return resource_ids
        if isinstance(resource_ids, str):
            with open(resource_ids, "r") as fd:
                return fd.read().splitlines()
        raise ValueError("input should be a list or a filename")

    async def _download(self, resource_id, session, folder):
        url = self.url_format.format(resource_id)
        path = Path(self.cache_dir / folder / resource_id)
        if path.is_file():
            return
        async with session.get(str(url)) as resp:
            with open(path, "wb") as fd:
                async for chunk in resp.content.iter_chunked(1024):
                    fd.write(chunk)

    async def _download_all(self):
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            await asyncio.gather(
                *[self._download(url, session, "responses") for url in self.pprox_ids],
                *[self._download(url, session, "stimuli") for url in self.wav_ids],
            )


class MemorySource(DataSource):
    """Loads data from given dictionaries"""

    def __init__(self, responses: dict, stimuli):
        self.responses = responses
        self.stimuli = stimuli

    def _get_raw_responses(self):
        return self.responses

    def get_stimuli(self):
        return self.stimuli


def _fix_pprox(responses: Dict[str, Pprox], durations: Dict[str, float]):
    for name, json_data in responses.items():
        if json_data.get("$schema") == "https://meliza.org/spec:2/stimtrial.json#":
            pass
        # if the pprox does not conform to the stimtrial spec,
        # we need to try to guess what format it is, and modify it
        # accordingly
        elif json_data.get("experiment") == "induction":
            _ar_data_shim(json_data, durations)
            _cn_data_shim(json_data, durations)
        elif json_data.get("protocol") in ["songs", "chorus"]:
            _cn_data_shim(json_data, durations)
        else:
            raise UnrecognizedPproxFormat(name)
        for i, trial in enumerate(json_data["pprox"]):
            trial["index"] = i
            trial["interval"] = tuple(trial["interval"])
            trial["stimulus"]["interval"] = tuple(trial["stimulus"]["interval"])
    return responses


class UnrecognizedPproxFormat(Exception):
    """Exception raised when the needed metadata could not be found in a pprox file"""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return (
            f"could not detect pprox format for file {self.name}, try making sure"
            " the pprox conforms to stimtrial and that '$schema' is set"
        )


def _ar_data_shim(json_data, durations):
    """reformat data from auditory-restoration project format to colony-noise project format"""
    for trial in json_data["pprox"]:
        _rename_key(trial, "event", "events")
        if trial.get("units") == "ms":
            del trial["units"]
            trial["events"] = [x / 1000 for x in trial["events"]]
        trial["stim"] = f"{trial['stimulus']}_{trial['condition']}"
        _rename_key(trial, "trial", "index")
        # we want to put the `interval` information in the cn_pprox format
        # so we list absolute recording start and stop with a sample rate of 1
        trial["recording"] = {
            "start": 0,
            "stop": trial["stim_on"] + durations[trial["stim"]] + 1,
        }
    json_data["entry_metadata"] = {"sampling_rate": 1}
    del json_data["experiment"]
    json_data["protocol"] = "songs"
    return json_data


def _cn_data_shim(json_data, durations):
    """reformat data from colony-noise project format to stimtrial format"""
    metadata = json_data["entry_metadata"]
    if isinstance(metadata, list):
        metadata = metadata[0]
    sampling_rate = metadata["sampling_rate"]
    del json_data["entry_metadata"]
    json_data["$schema"] = "https://meliza.org/spec:2/stimtrial.json#"
    del json_data["protocol"]
    for trial in json_data["pprox"]:
        offset = trial.get("offset") or 0
        trial["interval"] = (
            (trial["recording"]["start"] / sampling_rate) - offset,
            (trial["recording"]["stop"] / sampling_rate) - offset,
        )
        del trial["recording"]
        stim_off = durations[trial["stim"]]
        trial["stimulus"] = {
            "name": trial["stim"],
            "interval": (trial["stim_on"], trial["stim_on"] + stim_off),
        }
        del trial["stim"]
        del trial["stim_on"]
    return json_data


def _rename_key(obj, old_name, new_name):
    obj[new_name] = obj[old_name]
    del obj[old_name]
