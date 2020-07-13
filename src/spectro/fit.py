#!python3

"""
fit.py: Tools for fitting spectroscopic data.
"""
import numpy as np
import lmfit
import logging
from scipy.interpolate import interp1d
from decorator import decorator

logger = logging.getLogger(__name__)


def convolve(arr, kernel):
    """
    Convolution of array with kernel. The kernel is not normalized.
    """
    logger.debug("Convolving...")
    npts = min(len(arr), len(kernel))
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    #norm = np.sum(kernel)
    norm = 1
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts)/2)
    return out[noff:noff+npts]/norm


def gauss_kernel(x, t0, irf):
    """
    Gaussian convolution kernel.

    Parameters
    ----------
    x : array-like
        Independant variable
    t0 : array-like
        t0 offset
    irf : array-like
        Irf gaussian width (sigma)
    """
    midp = 0.5*(np.max(x)+np.min(x))
    return lmfit.lineshapes.gaussian(x, 1, midp+t0, irf)


def regrid(idx):
    """
    Decorator factory to compute a model on a constant grid, then interpolate.

    This is to be used for reconvolution fits when the independant axis isn't
    evently spaced. This function returns a decorator. You should call the
    result of this function with the model to regrid. The constant grid

    Parameters
    ----------
    idx : int
        Index of variable to regrid in client function.

    Returns
    -------
    regridder : decorator

    Example
    -------
    ```
    def model(x, amp, tau, t0, sig):
        # Convolution assumes constant grid spacing.
        return convolve(step(x)*exp_decay(x, amp, tau), gauss_kernel(x, t0, sig))

    deco = regrid(0)
    regridded = deco(model)
    # Or, on a single line
    regridded = regrid(0)(model) # compute on argument 0
    # Or, during definition
    @regrid(0)
    def model(x, *args):
        ...
    ```
    """
    logger.debug("Applying 'regrid' decorator")
    def _regrid(func, *args, **kw):
        logger.debug("Regridding func {}".format(func.__name__))
        x = args[idx]
        #print("regridding...")
        mn, mx = np.min(x), np.max(x)
        extension=1
        margin = (mx-mn)*extension
        dx = np.abs(np.min(x[1:]-x[:-1]))
        #print("regrid args", args)
        #print("regrid kw", kw)
        #print("regrid func", func)
        grid = np.arange(mn-margin, mx+margin+dx, dx)
        args = list(args)
        args[idx] = grid
        y = func(*args, **kw)
        #print("y", y)
        intrp = interp1d(grid, y, kind=3, copy=False, assume_sorted=True)
        return intrp(x)
    return decorator(_regrid)

def resid_vs(data):
    """
    Decorator factory to return the residuals of a function vs `data`.

    This permits the easy generation of residuals functions for fitting
    algorithms.

    Parameters
    ----------
    data : array-like
        Observation data. Should be compatible with the results from target
        function.

    Returns
    -------
    residual_decorator : decorator
        Decorator to compute residuals.

    Example:
    --------
    def model(x, amp, f, tau):
        return amp*np.sin(2*np.pi*x*f)*np.exp(-x/tau)

    x = np.linspace(0, 10, 100)
    y = model(x, 1, 0.5, 3) + 0.05 * np.random.randn(x.size)
    resid = resid_vs(y)(model)
    """
    def _resid(func, *args, **kw):
        y = func(*args, **kw)
        return y-data
    return decorator(_resid)

@decorator
def flat_out(f, *a, **kw):
    """Flatten the output of target function."""
    return f(*a, **kw).flatten()

def xpm(x, amp, x0, sigma):
    """
    Cross phase modulation artifact.

    Computed as the derivative of the gaussian function (not normalized):
    -amp*(x-x0)/sigma**2 * np.exp(-0.5*(x-x0)**2/sigma**2


    Parameters
    ----------
    """
    xr = x-x0
    sig2i = 1/(sigma*sigma)
    return -amp*xr*sig2i*np.exp(-xr*xr*sig2i*0.5)



def step(x):
    """Heaviside step function."""
    step = np.ones_like(x, dtype='float')
    step[x<0] = 0
    step[x==0] = 0.5
    return step

def make_global_multiexp(n_curves, n_exp):
    """
    Make a multiexponential global model for use with lmfit.

    This is suitable for DAS analysis. Contains exponential decays convolved
    with a gaussian IRF and a XPM artifact.

    Parameters
    ----------
    n_curves : int
        Number of curves in data.
    n_exp: int
        Number of exponential components.

    Returns
    -------
    global_model : function
        The global model function.

    Notes
    -----
    The returned model has the following arguments.
    params : dictionnary-like
        dict or lmfit.Parameters object representing the parameters
        It should have the following keys:
            t0 : time_zero
            irf : irf width (sigma)
            xpm_w : cross-phase modulation artifact width (sigma)
            tau_j : lifetimes. j in [0, n_exp[
            a_i_j : amplitudes. i in [0, n_curves[, j in [0, n_exp[
            xpma_i : amplitude of XPM. i in [0, n_curves[
    x : 1D array-like
        The independant variable (most likely time)

    The model returns an array with dimensions (n_curves, x.size).

    Example
    -------
    n_curves, n_exp = 4, 3
    model = regrid(1)(make_global_multiexp(n_curves, n_exp)
    resid = resid_vs(data)(model)
    guess = lmfit.Parameters()
    # ... populate the `guess`
    mini = lmfit.Minimizer(flat_out(resid), guess, fcn_args=(times,))
    results = mini.minimize()
    """
    def global_model(params, x):
        t0 = params["t0"]
        irf = params["irf"]
        xpm_w = params["xpm_w"]
        decays = np.zeros((n_exp, x.size))
        kern = gauss_kernel(x, t0, irf)
        m = x >= 0
        for i in range(n_exp):
            k = 'tau_{}'.format(i)
            decays[i, m] = step(x[m]) * np.exp(-x[m] / params[k])
            decays[i, :] = convolve(decays[i, :], kern)
        amps = np.array(
            [[params["a_{}_{}".format(i, j)]
              for j in range(n_exp)]
             for i in range(n_curves)]
        )
        res = np.dot(amps, decays)
        # add xpm
        xpm_artifact = xpm(x, 1, t0, xpm_w)
        xpm_amps = np.array([params["xpma_{}".format(i)]
                             for i in range(n_curves)])
        res += xpm_amps[:, np.newaxis] * xpm_artifact[np.newaxis, :]
        return res
    return global_model

def params_to_matrix(params, n_curves, n_exp, fmt="a_{}_{}"):
    """
    Convert fit parameters a_i_j to coefficient matrix.
    """
    das_mat = np.zeros((n_curves, n_exp))
    for i in range(n_curves):
        for j in range(n_curves):
            das_mat[i,j] = params[fmt.format(i,j)]
    return das_mat
