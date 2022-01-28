"""File I/O methods"""
import json
import asyncio
import aiohttp
from glob import glob
import parse
from scipy.io import wavfile
from urllib.parse import urljoin, urlparse
from appdirs import user_cache_dir
from abc import ABC, abstractmethod
from pathlib import Path
from decoding import appname, appauthor
from . import dataset

class DataSource(ABC):
    """Abstract class for data source

    Provides a common interface for concrete classes,
    which will define how to identify data (e.g. local path, URL)
    """
    @abstractmethod
    def get_raw_responses(self):
        """returns dictionary mapping names of files to pprox data

            pprox may be in the incorrect format
        """

    def get_responses(self):
        """returns dictionary mapping names of files to stimtrial pprox data
        """
        responses = self.get_raw_responses()
        stimuli = self.get_stimuli()
        durations = {name: len(s)/fs for name, (fs, s) in stimuli.items()}
        return _fix_pprox(responses, durations)

    @abstractmethod
    def get_stimuli(self):
        """returns dictionary mapping names of files to (sample_rate, samples) tuples
        """

    def __eq__(self, other) -> bool:
        return (
                (self.get_responses() == other.get_responses())
                and
                (self.get_stimuli() == other.get_stimuli())
                )


class FsSource(DataSource):
    """Loads data from local File System
    """
    def __init__(self, pprox_path_format, wav_path_format, stimuli_names=None, cluster_list=None):
        """
            load pprox data

            pprox_path_format: e.g. 'pprox/P120_1_1_{}.pprox'
            wav_path_format: e.g. 'wav/{}.wav'
            cluster_list: optional list of identifiers (that fill in
                the {} in `pprox_path_format`) to load from the path format.
                If not provided, load all matching files.
            stimuli_names: optional list of stimuli identifiers to load.
                Loads all matching files if not provided.
        """
        if not isinstance(pprox_path_format, str):
            pprox_path_format = str(pprox_path_format)
        if not isinstance(wav_path_format, str):
            wav_path_format = str(wav_path_format)
        self.pprox_path_format = pprox_path_format
        self.cluster_list = cluster_list
        self.wav_path_format = wav_path_format
        self.stimuli_names = stimuli_names
        self.get_stimuli = dataset.mem.cache(self.get_stimuli)

    def get_raw_responses(self):
        return load_pprox(
                self.pprox_path_format,
                cluster_names=self.cluster_list,
        )

    def get_stimuli(self):
        return load_stimuli(self.wav_path_format, self.stimuli_names)

class NeurobankSource(FsSource):
    """Downloads data from Neurobank
    """
    DOWNLOAD_FORMAT = 'resources/{}/download'
    def __init__(self, neurobank_registry, pprox_ids, wav_ids):
        """
            neurobank_registry: URL
            pprox_ids: list of resource IDs, or path to file containing such a list
            wav_ids: list of resource IDs, or path to file containing such a list
        """
        self.url_format = urljoin(neurobank_registry, self.DOWNLOAD_FORMAT)
        self.pprox_ids = self._get_list(pprox_ids)
        self.wav_ids = self._get_list(wav_ids)
        parsed_url = urlparse(neurobank_registry)
        self.cache_dir = Path(user_cache_dir(appname, appauthor)) / parsed_url.netloc
        responses_dir = self.cache_dir / 'responses'
        stimuli_dir = self.cache_dir / 'stimuli'
        responses_dir.mkdir(parents=True, exist_ok=True)
        stimuli_dir.mkdir(parents=True, exist_ok=True)
        asyncio.run(self._download_all())
        super().__init__(
                responses_dir / '{}',
                stimuli_dir / '{}',
                cluster_list=pprox_ids,
                stimuli_names=wav_ids,
        )

    @staticmethod
    def _get_list(resource_ids):
        if isinstance(resource_ids, list):
            return resource_ids
        if isinstance(resource_ids, str):
            with open(resource_ids, 'r') as fd:
                return fd.read().split('\n')
        raise ValueError("input should be a list or a filename")

    async def _download(self, resource_id, session, folder):
        url = self.url_format.format(resource_id)
        path = Path(self.cache_dir / folder / resource_id)
        if path.is_file():
            return
        async with session.get(str(url)) as resp:
            with open(path, 'wb') as fd:
                async for chunk in resp.content.iter_chunked(1024):
                    fd.write(chunk)

    async def _download_all(self):
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            await asyncio.gather(
                    *[self._download(url, session, 'responses') for url in self.pprox_ids],
                    *[self._download(url, session, 'stimuli') for url in self.wav_ids]
            )

class MemorySource(DataSource):
    """Loads data from given dictionaries
    """
    def __init__(self, responses, stimuli):
        self.responses = responses
        self.stimuli = stimuli

    def get_raw_responses(self):
        return self.responses

    def get_stimuli(self):
        return self.stimuli

def load_pprox(path_format, cluster_names=None, durations=None):
    """Load pprox files into a dictionary
        path_format: path containing "{}" which will be globbed
                     for all matching files and then parsed as
                     JSON
        cluster_names (iterable): if specified, load only the clusters
                       with the given names
        durations: dictionary of stim name -> stim duration in case
                    this info is not included in the pprox file
        returns: dictionary mapping the part of the filenames
                 represented by "{}" to their parsed contents
    """
    clusters = {}
    for name, path in _get_filenames(path_format, cluster_names):
        with open(path, 'r') as pprox_file:
            json_data = json.load(pprox_file)
            clusters[name] = json_data
    return clusters

def load_stimuli(path_format, stimuli_names=None):
    """Load wav files into a dictionary

        stimuli_names: iterable of strings
        path_format: path containing "{}" which will be replaced by
                     each of `stimuli_names`

        returns: dictionary mapping stimuli_names to (sample_rate, samples)
    """
    stimuli = {}
    for name, filename in _get_filenames(path_format, stimuli_names):
        sample_rate, samples = wavfile.read(filename)
        stimuli[name] = (sample_rate, samples)
    return stimuli

def _get_filenames(path_format, names):
    assert path_format.find('{}') != -1, (
            'path_format should include empty braces where a wildcard'
            ' would go.\n'
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

def _fix_pprox(responses, durations):
    for json_data in responses.values():
        if json_data.get('$schema') == "https://meliza.org/spec:2/stimtrial.json#":
            pass
        # if the pprox does not conform to the stimtrial spec,
        # we need to try to guess what format it is, and modify it
        # accordingly
        elif json_data.get('experiment') == 'induction':
            _ar_data_shim(json_data, durations)
            _cn_data_shim(json_data, durations)
        elif json_data.get('protocol') == 'songs':
            _cn_data_shim(json_data, durations)
        else:
            raise UnrecognizedPproxFormat
        for i, trial in enumerate(json_data['pprox']):
            trial['stim'] = trial['stimulus']['name']
            trial['index'] = i
    return responses


class UnrecognizedPproxFormat(Exception):
    def __str__(self):
        return "could not detect pprox format, try making sure the pprox conforms to stimtrial and that '$schema' is set"


def _ar_data_shim(json_data, durations):
    '''reformat data from auditory-restoration project format to colony-noise project format'''
    for trial in json_data['pprox']:
        _rename_key(trial, 'event', 'events')
        if trial.get('units') == 'ms':
            del trial['units']
            trial['events'] = [x / 1000 for x in trial['events']]
        _rename_key(trial, 'stim_uuid', 'stim')
        _rename_key(trial, 'trial', 'index')
        trial['recording'] = {
                'start': 0,
                'stop': trial['stim_on'] + durations[trial['stim']] + 1,
                }
    json_data['entry_metadata'] = {'sampling_rate': 1}
    del json_data['experiment']
    json_data['protocol'] = 'songs'
    return json_data

def _cn_data_shim(json_data, durations):
    '''reformat data from colony-noise project format to stimtrial format'''
    metadata = json_data['entry_metadata']
    if isinstance(metadata, list):
        metadata = metadata[0]
    sampling_rate = metadata['sampling_rate']
    del json_data['entry_metadata']
    json_data['$schema'] = "https://meliza.org/spec:2/stimtrial.json#"
    del json_data['protocol']
    for trial in json_data['pprox']:
        trial['interval'] = [
                trial['recording']['start'] / sampling_rate,
                trial['recording']['stop'] / sampling_rate
        ]
        del trial['recording']
        stim_off = durations[trial['stim']]
        trial['stimulus'] = {
                'name': trial['stim'],
                'interval': [trial['stim_on'], trial['stim_on'] + stim_off]
        }
        del trial['stim']
        del trial['stim_on']
    return json_data

def _rename_key(obj, old_name, new_name):
    obj[new_name] = obj[old_name]
    del obj[old_name]
