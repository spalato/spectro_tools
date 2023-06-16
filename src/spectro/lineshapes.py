import numpy as np
from lmfit.lineshapes import voigt
from scipy.special import assoc_laguerre, factorial, wofz, erfc

from spectro.utils import sq2, sq2dsqpi, sqpi


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
