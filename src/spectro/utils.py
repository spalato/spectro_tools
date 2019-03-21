"""
spectro.utils : Generic tools to be used in the analysis of spectroscopic
data.
"""

import numpy as np
from scipy.special import assoc_laguerre, factorial, erfc, wofz
from lmfit.lineshapes import voigt
from scipy.constants import eV, h, c, nano, femto, pi
from math import sqrt, pi
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.optimize import brentq

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
    """Divide by max absolute value"""
    return x / np.max(np.abs(x))

def fc_factor(m, n, s):
    """
    Franck-Condon overlap between states with identical vibrational frequency.
    
    Parameters
    ----------
    m : positive int
        Vibrational quantum number in initial state
    n : positive int
        Vibrational quantum number in final state
    S : positive float
        Huang Rhys parameter
    """
    #
    #if any([v<0 for v in [m, n, s]]):
    #    raise ValueError("Vibrational quantum numbers and Huang-Rhys must be positive")

    n = np.asarray(n, dtype='float64')
    m = np.asarray(m, dtype='float64')
    s = np.asarray(s, dtype='float64')
    n, m = np.meshgrid(n, m)
    # swap n, m such that n>=m. Otherwise our assoc_laguerre spits a nan.
    d = n-m
    n_ = np.where(n>=m, n, m)
    m_ = np.where(n>=m, m, n)

    lag = assoc_laguerre(s, m_, np.abs(d))
    f = factorial(m_)/factorial(n_)
    assert np.all(f>0)
    #return np.exp(-s)*np.power(s,d)*f*lag*lag
    return np.exp(-s)*np.power(s,np.abs(d))*f*lag*lag


def vibronic_intensity(m, n, s, e_vib, kt=0):
    """
    Intensity of a Franck-Condon transition

    Parameters
    ----------
    m : array-like, int
        Vibrational quantum number in the initial manifold
    n : array-like, int
        Vibrational quantum number in the final manifold
    s : float
        Huang-Rhys factor S
    e_vib : float
        Vibrational energy
    kt : float
        Thermal energy

    Returns
    -------
    intensities: array-like, float
        Intensity of the vibrational bands
    """
    # compute boltzmann factors
    boltz_f = np.exp(-m * e_vib / kt) if kt > 0 else [1]
    boltz_f /= np.sum(boltz_f)
    # FC factors
    fcf = fc_factor(m, n, s)
    fcf *= boltz_f[:, np.newaxis]
    return fcf



def vibronic_ls(x, s, sigma, gamma, e_vib,  kt=0, n_max=None, m_max=None):
    """
    Produce a vibronic (Frank-Condom) lineshape.
    
    The vibronic transition amplitude computed relative to 0 (ie: relative to 
    the electronic transition energy). Lines are broadened using a voigt
    profile.
    
    Parameters
    ----------
    x : np.ndarray
        Energy values. x==0 is the 0->0 line (no vibrational quanta change)
    s : float
        Huang-Rhys parameter S
    e_vib : float
        Energy of a vibrational quanta
    sigma : float
        Width (1/e^2) of gaussian component
    gamma : float
        Width of Lorententzian component
    kt : float
        Thermal energy. If >0, will compute transitions from vibrationally
        excited states. Default 0.
    n_max : int
        Largest vibrational number in final manifold. If not supplied, a guess 
        is provided, but may not be adequate.
    m_max : int
        Largest vibrational number in orginal manifold. If not supplied, a guess
        is provided, but may not be adequate.
    """
    #determine n, m, values
    if m_max is None:
        m_max = 0 if kt==0 else int(kt/e_vib*10) # found that factor with my thumb
    if n_max is None:
        n_max = m_max + int(10*s)
    n = np.arange(n_max+1)
    m = np.arange(m_max+1)
    # compute boltzmann factors
    #boltz_f = np.exp(-m*e_vib/kt) if kt>0 else [1]
    #boltz_f /= np.sum(boltz_f)
    # FC factors
    #fcf = fc_factor(m, n, s)
    #fcf *= boltz_f[:,np.newaxis]
    fcf = vibronic_intensity(m, n, s, e_vib, kt)
    n, m = np.meshgrid(n, m)
    dvib = n-m
    y = np.zeros_like(x)
    for d, f in zip(dvib.flatten(), fcf.flatten()):
        y += voigt(x, f, d*e_vib, sigma, gamma)
    return y


def vibronic_emission(x, amp, x0, s, sigma, gamma, e_vib,  kt=0, **kw):
    """
    Produce a vibronic (Frank-Condom) lineshape.
    
    The vibronic emission lineshape. Lines are broadened using a voigt profile.
    
    Parameters
    ----------
    x : np.ndarray
        Energy values.
    amp : float
        Transition amplitude.
    x0 : float
        Electronic transition energy. (zero-phonon line)
    s : float
        Huang-Rhys parameter S
    e_vib : float
        Energy of a vibrational quanta
    sigma : float
        Width (1/e^2) of gaussian component
    gamma : float
        Width of Lorententzian component
    kt : float
        Thermal energy. If >0, will compute transitions from vibrationally
        excited states. Default 0.
    n_max : int
        Largest vibrational number in final manifold. If not supplied, a guess 
        is provided, but may not be adequate.
    m_max : int
        Largest vibrational number in orginal manifold. If not supplied, a guess
        is provided, but may not be adequate.
    """
    return amp*vibronic_ls(-x+x0, s, sigma, gamma, e_vib, kt=kt, **kw)


def vibronic_absorption(x, amp, x0, s, sigma, gamma, e_vib,  kt=0, **kw):
    """
    Produce a vibronic (Frank-Condom) lineshape.
    
    Vibronic absorption lineshape. Lines are broadened using a voigt profile.
    
    Parameters
    ----------
    x : np.ndarray
        Energy values.
    amp : float
        Transition amplitude.
    x0 : float
        Electronic transition energy. (zero-phonon line)
    s : float
        Huang-Rhys parameter S
    e_vib : float
        Energy of a vibrational quanta
    sigma : float
        Width (1/e^2) of gaussian component
    gamma : float
        Width of Lorententzian component
    kt : float
        Thermal energy. If >0, will compute transitions from vibrationally
        excited states. Default 0.
    n_max : int
        Largest vibrational number in final manifold. If not supplied, a guess 
        is provided, but may not be adequate.
    m_max : int
        Largest vibrational number in orginal manifold. If not supplied, a guess
        is provided, but may not be adequate.
    """
    return amp*vibronic_ls(x-x0, s, sigma, gamma, e_vib, kt=kt, **kw)


def bloch_reph_diag(x, amp, x0, s, g):
    """
    Rephasing bloch diagonal lineshape

    Parameters
    ----------
    x : array-like
        Independant axis (energy/frequency)
    amp : float
        Amplitude
    x0 : float
        Peak position
    s : float
        Inhomogeneous linewidth sigma
    g : float
        Homogeneous linewidth gamma

    Returns
    -------
    lineshape : array-like
        Complex lineshape amplitude
    """
    return np.sqrt(2*np.pi)/g*voigt(x, amp, x0, s, g)


def bloch_reph_adiag(x, amp, x0, s, g):
    """
    Rephasing bloch anti-diagonal lineshape

    Parameters
    ----------
    x : array-like
        Independant axis (energy/frequency)
    amp : float
        Amplitude
    x0 : float
        Peak position
    s : float
        Inhomogeneous linewidth sigma
    g : float
        Homogeneous linewidth gamma

    Returns
    -------
    lineshape : array-like
        Complex lineshape amplitude
    """
    xs = g-1j*(x-x0)
    G = xs/s/sq2
    y = wofz(1j*G)/(s*xs)
    return amp*y


def bloch_nonreph_diag(x, amp, x0, s, g):
    """
    Non-rephasing bloch diagonal lineshape. Not normalized

    Parameters
    ----------
    x : array-like
        Independant axis (energy/frequency)
    amp : float
        Amplitude
    x0 : float
        Peak position
    s : float
        Inhomogeneous linewidth sigma
    g : float
        Homogeneous linewidth gamma

    Returns
    -------
    lineshape : array-like
        Complex lineshape amplitude
    """
    """G = (g+1j*(x-x0))/np.sqrt(2)/s ## TODO: change to use wofz
    y = 2/s**2*(1-np.sqrt(np.pi)*G*np.exp(G**2)*erfc(G))
    # the imaginary part needs to be flipped by -1
    return np.conj(y)/np.sqrt(2*np.pi)*amp"""
    G = (g+1j*(x-x0))/s/sq2
    return sq2dsqpi/s/s*(1-sqpi*G*wofz(1j*G))*amp


def _bloch_na_limit(s, g):
    gp = g/np.sqrt(2)/s
    gpgp = gp*gp
    return -1/np.sqrt(2)/s*np.exp(gpgp)*(2*gp*erfc(gp) - 2/np.sqrt(np.pi)*np.exp(-gpgp))


def bloch_nonreph_adiag(x, amp, x0, s, g):
    """
    Non-rephasing bloch antidiagonal lineshape

    Parameters
    ----------
    x : array-like
        Independant axis (energy/frequency)
    amp : float
        Amplitude
    x0 : float
        Peak position
    s : float
        Inhomogeneous linewidth sigma
    g : float
        Homogeneous linewidth gamma

    Returns
    -------
    lineshape : array-like
        Complex lineshape amplitude
    """
    xr = x-x0
    G = (xr+1j*g)/np.sqrt(2)/s
    l = wofz(G).imag/xr
    missing = np.isnan(l)
    if any(missing):
        m = np.abs(xr/s) < 1E-12
        l[m] = _bloch_na_limit(s, g)
    return amp/s*l


def bloch_abs_diag(x, amp, x0, s, g, nmax=True):
    y = (bloch_reph_diag(x, 0.5, x0, s, g)
         + bloch_nonreph_diag(x, 0.5, x0, s, g))
    if nmax:
        y /= np.max(np.abs(y))
    return y*amp


def bloch_abs_adiag(x, amp, x0, s, g, nmax=True):
    y = (bloch_reph_adiag(x, 0.5, x0, s, g)
         + bloch_nonreph_adiag(x, 0.5, x0, s, g))
    if nmax:
        y /= np.max(np.abs(y))
    return y*amp


def brent_fwhm(t, amp, retall=False):
    """Obtain FWHM using brent's method to find half max crossings.

    This will not behave well if more than two halfmax crossings are present.
    """
    assert np.all(np.diff(t) > 0)
    amp = amp - np.min(amp)
    amp /= np.max(amp)
    ampf = UnivariateSpline(t, amp-0.5, k=1, s=0)
    t_max = t[np.argmax(amp)]
    t1 = brentq(ampf, t[0], t_max)
    t2 = brentq(ampf, t_max, t[-1])
    if retall:
        return t2-t1, (t1, t2)
    else:
        return t2-t1

def brent_fwtm(t, amp, retall=False):
    """Obtain FWHM using brent's method to find half max crossings.

    This will not behave well if more than two halfmax crossings are present.
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