# spectro.io: Loading and saving data

import numpy as np
from itertools import dropwhile


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
