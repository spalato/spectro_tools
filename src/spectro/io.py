# spectro.io: Loading and saving data

import numpy as np


def load_tcspc_dat(fname):
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

