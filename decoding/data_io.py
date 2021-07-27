"""File I/O methods"""
import json
from glob import glob
from parse import parse
from scipy.io import wavfile

def load_pprox(path_format):
    """Load pprox files into a dictionary
        path_format: path containing "{}" which will be globbed
                     for all matching files and then parsed as
                     JSON

        returns: dictionary mapping the part of the filenames
                 represented by "{}" to their parsed contents
    """
    clusters = {}
    for path in glob(path_format.format("*")):
        with open(path, 'r') as pprox_file:
            cluster_name = parse(path_format, path)[0]
            clusters[cluster_name] = json.load(pprox_file)
    return clusters

def load_stimuli(path_format, stimuli_names):
    """Load wav files into a dictionary

        stimuli_names: iterable of strings
        path_format: path containing "{}" which will be replaced by
                     each of `stimuli_names`

        returns: dictionary mapping stimuli_names to (sample_rate, samples)
    """
    stimuli = {}
    for name in stimuli_names:
        sample_rate, samples = wavfile.read(path_format.format(name))
        stimuli[name] = (sample_rate, samples)
    return stimuli
