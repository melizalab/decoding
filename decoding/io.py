"""File I/O methods"""
import json
from glob import glob
from parse import parse
from scipy.io import wavfile

def load_pprox(path_format, cluster_names=None):
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
    for name, path in _get_filenames(path_format, cluster_names):
        with open(path, 'r') as pprox_file:
            clusters[name] = json.load(pprox_file)
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
    if names is None:
        filenames = glob(path_format.format("*"))
        names = map(lambda p: parse(path_format, p)[0], filenames)
    else:
        filenames = map(path_format.format, names)
    return zip(names, filenames)