import numpy as np
from scipy.special import assoc_laguerre, factorial
from lmfit.lineshapes import voigt
import matplotlib.pyplot as plt

# Normally, all these functions are in a module called `spectro.utils`. 
# I would only perform the following import:
# from spectro.utils import vibronic_emission


# These functions nromally are in a module. As such, I made them a bit "safer"
# by performing checks and conversions on the arguments, as well as some
# basic asserts. This is not strictly necessary.
# The overall function (vibronic_emission) is broken up in many small steps.
# This looks a bit silly, but I ended up needing the separate bits at various
# points. Breaking things up in very small functions is usually good practice.
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


x = np.linspace(1.8, 2.5, 1000)
e0 = 2.17
s = 0.5
sigma = 0.01
gamma = 0.001
e_vib = 0.07
y1 = vibronic_emission(x, 1, e0, s, sigma, gamma, e_vib, 0)
y2 = vibronic_emission(x, 1, e0, s, sigma, gamma, e_vib, 0.025)
y3 = vibronic_emission(x, 1, e0, s, sigma, gamma, e_vib, 0.2)
plt.figure()
plt.plot(x, y1, label="kT=0")
plt.plot(x, y2, label="kT=RT")
plt.plot(x, y3, label="kT=200 meV")
plt.legend()
plt.savefig("fc_emission.png", dpi=150)
