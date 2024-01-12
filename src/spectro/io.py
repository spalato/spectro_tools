# spectro.io: Loading and saving data

import numpy as np
from itertools import dropwhile
import logging
logger = logging.getLogger(__name__)


def read_tcspc_dat(fname):
    """Load a picoquant TCSPC histogram in `.dat` format.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    step : (M,) np.ndarray
        ns/bin
    data : (N, M) np.ndarray
    """
    with open(fname) as f:
        # go through the first lines
        for i in range(8):
            f.readline()
        # get the steps
        steps = np.array([float(e) for e in f.readline().strip().split()])
        # dump next line
        f.readline()
        # load histogram data
        data = np.loadtxt(f)
    # return and ensure data has 2 dim
    return steps, data.reshape((-1, 1)) if data.ndim==1 else data


def read_oceanview(fname):
    """Read spectra from saved using OceanView

    Parameters
    ----------
    fname: str
        File name

    Returns
    -------
    wl : np.ndarray
        Wavelength
    counts : np.ndarray
        Signal
    """
    with open(fname) as f:
        ln = next(
            dropwhile(
                lambda ln: "Pixels in Spectrum" not in ln, 
                f
            )
        ).strip().split(" ")
        npix = int(ln[-1])
        dat = np.loadtxt(dropwhile(lambda ln: "Begin Spectral Data" not in ln, f), skiprows=1)
        assert dat.shape[1] == 2 # not sure how multiple columns are handled
        return dat[:,0], dat[:,1]


def read_asc(fname, delimiter=","):
    """
    Read a single `asc` file, the ASCII format from Andor Solis.

    Parameters
    ----------
    fname : str, path-like
        File to open.

    """
    logger.debug("Loading `.asc` file: %s", fname)
    with open(fname) as f:
        contents = f.read()
    meta_start = contents.find("Date and Time")
    logger.debug("  Metadata at %i", meta_start)
    if meta_start == 0:
        start = contents.find("\n"*3)
        end = None
    else:
        start = None
        end = contents.find("\n"*3)
    return np.loadtxt((ln for ln in contents[start:end].splitlines() if ln), delimiter=delimiter)


def read_ta_dat(fname, sort_t=True):
    "Read a `.dat` file from OMAFEMTO."
    dat = np.loadtxt(fname)
    t = dat[0,1:]
    wl = dat[1:,0]
    z = dat[1:,1:]
    if sort_t:
        # This is overkill. We should just reverse
        srt_idx = np.argsort(t)
        t = t[srt_idx]
        z = z[:,srt_idx]
    return t, wl, z


def read_ab_dat(fname):
    "Read a `.dat` file from OMAFEMTO for absorption spectrum"
    dat = np.loadtxt(fname)
    wl = dat[1:,0]
    z = dat[1:,1:]
    return wl, z
