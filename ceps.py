import os
import glob
import numpy as np
import scipy
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

from utils import DATASET_DIR, GENRE_LIST


def write_ceps(ceps, fn):
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print "Written ", data_fn


def create_ceps(fn):
    """
        Creates the MFCC features. 
    """    
    sample_rate, x = scipy.io.wavfile.read(fn)
    x[x == 0] = 1
    ceps, mspec, spec = mfcc(x)
    write_ceps(ceps, fn)


def read_ceps(genre_list, base_dir=DATASET_DIR):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    x = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            num_ceps = len(ceps)
            x.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
            y.append(label)
    return np.array(x), np.array(y)


def create_ceps_test(fn):
    """
        Creates the MFCC features from the test files,
        saves them to disk, and returns the saved file name.
    """
    sample_rate, x = scipy.io.wavfile.read(fn)
    x[x == 0] = 1
    np.nan_to_num(x)
    ceps, mspec, spec = mfcc(x)
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print "Written ", data_fn
    return data_fn


def read_ceps_test(test_file):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    x = []
    y = []
    ceps = np.load(test_file)
    num_ceps = len(ceps)
    x.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    return np.array(x), np.array(y)


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    traverse = []
    for subdir, dirs, files in os.walk(DATASET_DIR):
        traverse = list(set(dirs).intersection(set(GENRE_LIST)))
        break
    print "Working with these genres --> ", traverse
    print "Starting ceps generation"     
    for subdir, dirs, files in os.walk(DATASET_DIR):
        for fl in files:
            path = subdir+'/'+fl
            if path.endswith("wav"):
                tmp = subdir[subdir.rfind('/', 0)+1:]
                if tmp in traverse:
                    create_ceps(path)
                    
    stop = timeit.default_timer()
    print "Total ceps generation and feature writing time (s) = ", (stop - start) 
