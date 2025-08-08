"""
spectro.utils : Generic tools to be used in the analysis of spectroscopic
data.
"""

import numpy as np
from scipy.constants import eV, h, c, nano, femto, pi
from math import sqrt, pi
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.optimize import brentq
from scipy.ndimage import median_filter

sq2 = sqrt(2)
sqpi = sqrt(pi)
sq2dsqpi = sq2/sqpi


def nm2ev(x):
    return h*c/x/nano/eV

def ev2nm(x):
    return h*c/x/eV/nano

def ev2phz(x):
    scale = eV*femto/h
    return x*scale

def ev2angphz(x):
    return ev2phz(x)*2*pi

def phz2ev(x):
    scale = h/eV/femto
    return x*scale

def angphz2ev(x):
    return phz2ev(x/2/pi)

def phz2nm(x):
    return ev2nm(phz2ev(x))

def nm2phz(x):
    return ev2phz(nm2ev(x))

def nm2angphz(x):
    return nm2phz(x)*2*pi

def angphz2nm(x):
    return phz2nm(x/2/pi)

def argnear(a, v):
    """Return index of value in `a` closest to `v`"""
    return np.argmin(np.abs(a-v))

def nearest(a, v):
    """Return value in `a` closest to `v`"""
    return a[argnear(a, v)]

def between(a, l, u):
    """Boolean mask for values in `a` larger than `l`, smaller than `u`"""
    return (a > l) & (a < u)

def band(a, v, w):
    """Boolean mask for values in `a` in a band centered at `v` of width `w`."""
    hw = w/2
    return between(a, v-hw, v+hw)

def project(data, mask, axis=None):
    """Project a slice of `data` where `mask` is True along `axis`."""
    return data.compress(mask, axis=axis).sum(axis=axis)

def regularize(x, upsample=2):
    if x.ndim != 1: raise ValueError("Built for 1d axis")
    return np.linspace(np.min(x), np.max(x), x.size * upsample)


def _intrp2d_cmplx(x, y, z, xg, yg):
    re = RectBivariateSpline(x, y, z.real)
    im = RectBivariateSpline(x, y, z.imag)
    return re(xg, yg) + 1j * im(xg, yg)


def _intrp2d_real(x, y, z, xg, yg):
    intrp = RectBivariateSpline(x, y, z)
    return intrp(xg, yg)

def prepare_axis(x):
    x = np.asarray(x)
    dx = np.diff(x)
    if np.all(dx > 0):
        idx = slice(None, None, 1)
    elif np.all(dx < 0):
        idx = slice(None, None, -1)
    else:
        # could implement using argsort
        raise ValueError("array must be monotonous.")
    return x[idx], idx


def regrid2d(x, y, z, upsample=2):
    x, x_idx = prepare_axis(x.copy())
    y, y_idx = prepare_axis(y.copy())
    z = z.copy()[x_idx, y_idx]
    x_reg = regularize(x, upsample)
    y_reg = regularize(y, upsample)
    if np.iscomplexobj(z):
        intrp = _intrp2d_cmplx
    else:
        intrp = _intrp2d_real
    return x_reg, y_reg, intrp(x, y, z, x_reg, y_reg)


def rescale(x):
    """Divide by max absolute value. Non finite values are ignored."""
    return x / np.max(np.abs(x), initial=0, where=np.isfinite(x))


def extrema(x):
    """Return the max or min of x, whichever has the largest absolute value."""
    amax = np.max(np.abs(x))
    max_ = np.max(x)
    return max_ if max_ == amax else np.min(x)


def sparkline(x):
    """Shift and scale data to the 0-1 range."""
    return rescale(x-np.min(x))


def brent_fwhm(x, amp, shift=True):
    """Obtain FWHM using brent's method to find half max crossings.

    This algoritms works by scaling the data by its max amplitude, then finding
    the points where: 0 = y_s - 0.5. By default, it shifts and scale the data 
    to the [0, 1] interval. The minimum value can be left unchanged if 
    `shift=False`.

    This will not behave well if more than two halfmax crossings are present:
    two of them will be arbitarily selected.

    Parameters
    ---------
    x : (N,) np.ndarray
        Independant variable, in ascending order
    amp : (N,) np.ndarray
        Dependant variable
    shift : bool, default True
        Shift minimum to 0.

    Returns
    -------
    fwhm : float
        Full width at half maximum
    (x1, x2) : (float, float), only if ret_pts
        Half max crossing positions.
    """
    assert np.all(np.diff(x) > 0)
    if shift:
        amp = amp - np.min(amp)
    amp = amp/np.max(amp)
    ampf = UnivariateSpline(x, amp-0.5, k=1, s=0)
    t_max = x[np.argmax(amp)]
    t1 = brentq(ampf, x[0], t_max)
    t2 = brentq(ampf, t_max, x[-1])
    return t2-t1, (t1, t2)


def largest_fwhm(x, y, retall=False):
    y = y.copy()
    y -= np.min(y)
    y /= np.max(y)
    g1 = x[np.argmax(y > 0.5)]
    g2 = x[::-1][np.argmax(y[::-1] > 0.5)]
    yf = UnivariateSpline(x, y-0.5, k=1, s=0)
    x1 = brentq(yf, x[0], g1)
    x2 = brentq(yf, g2, x[-1])
    if retall:
        return x2-x1, (x1, x2)
    else:
        return x2-x1


def brent_fwtm(t, amp, retall=False):
    """Obtain FWHM using brent's method to find 1/10 max crossings.

    This will not behave well if more than two crossings are present.
    """
    amp = amp - np.min(amp)
    amp /= np.max(amp)
    ampf = UnivariateSpline(t, amp-0.1, k=1, s=0)
    t_max = t[np.argmax(amp)]
    t1 = brentq(ampf, t[0], t_max)
    t2 = brentq(ampf, t_max, t[-1])
    if retall:
        return t2-t1, (t1, t2)
    else:
        return t2-t1

def unpack_trace(data):
    """
    Extract axes and data from a packed data matrix.

    Returns
    -------
    t     (M,) np.ndarray
    wl    (N,) np.ndarray
    trace (M, N) np.ndarray
    """
    wl = data[1:,0]
    t = data[0,1:]
    trace = data[1:,1:]
    return t, wl, trace


def pack_trace(t, wl, trace):
    """
    Pack axes and trace as a single array.

    Parameters
    ----------
    t : (M,) np.ndarray
        Times arrays
    wl : (N,) np.ndarray
        Wavelength arrays
    trace : (M,N) np.ndarray
        Intensity trace
    """
    packed = np.empty([i+1 for i in trace.shape])
    packed.fill(np.nan)
    packed[1:,1:] = trace
    packed[1:,0] = wl
    packed[0, 1:] = t
    return packed


def despike_median(data, size, threshold=5):
    cutoff = np.std(data) * threshold
    filtered = median_filter(data, size=size)
    reject = np.abs(filtered-data) > cutoff
    despiked = np.copy(data)
    despiked[reject] = filtered[reject]
    return despiked

