"""
cumulant.py: build lineshapes using the cumulant expansion (AKA Mukamel)

Truncated to second order of course.

The various integration steps are carried out using simpson numerical
integration on a constant grid. Grid parameters are very important! Most
integrands look either like lineshapes (in the frequency domain) or damped
harmonic oscillators (in time). In both cases, the integration grid needs to
be long enough to avoid truncation artifacts and fine enough to properly sample
the integrand. The integrands can be inspected directly using
`make_g(f).integrand` and `_abs_integrand` for the lineshape function and the
absorption lineshape, respectively.

Absorption spectra using Markovian Brownian Oscillator and Lorentz lineshape
required some fooling around with parameter to get the proper scale of the
Huang-Rhys. This should be reviewed.
"""
# Long live numerical integration

import numpy as np
from scipy.integrate import simps

twopi = 2*np.pi


def coth(*a, **kw):
    """Hyperbolic cotangent"""
    return 1/np.tanh(*a, **kw)


def g_homo(t, gamma):
    """
    Lineshape function for the homogeneous limit: gamma*t

    Parameters
    ----------
    t : array-like float
        time
    gamma : float
        Inverse dephasing time 1/T_2^\star
    """
    return t*gamma


def g_inhomo(t, sigma):
    """
    Lineshape function for the inhomogeneous limit: sigma**2*t**2/2

    Parameters:
    -----------
    t : array-like float
        time
    sigma : float
        Frequency distribution \Delta \omega
    """
    return 0.5*sigma**2*t**2


def g_kubo(t, tau, sigma):
    """
    Lineshape function for the Kubo ansatz

    Parameters:
    -----------
    t : array-like float
        time
    tau : float
        Timescale for dephasing
    sigma : float
        Frequency distribution \Delta \omega
    """
    return sigma**2*tau**2*(np.exp(-t/tau)+t/tau-1)


def g_huang_rhys(t, omega_0, kt):
    """
    Lineshape function for undamped displaced vibrational oscillator.

    """
    w0t = omega_0*t
    return 1j*omega_0*t+coth(omega_0/kt/2)*(1-np.cos(w0t))+ 1j*(np.sin(w0t)-w0t)

def make_g(c):
    """
    Generate lineshape function g(t) from the correlation function C''(omega)

    Generate the lineshape function g(t) from the odd part of the spectral
    correlation function \tilde{C}''(\omega). This is achieved by numerical
    simpson integration of `c` over a grid. The grid can be supplied to `g`.


    Parameters
    ----------
    c : callable, c(omega, *c_a, **c_kw)
        Odd part of the correlation function. First argument must be angular
        frequency.

    Returns
    -------
    g : callable
        Lineshape function g(t).
        Arguments:
        t : array-like floats
            time values
        kt : float
            Thermal energy.
        Keywords:
        lim : float
            Limits of the (symmetric) default grid. Ignored if `wgrid` is
            supplied. Defaults to 20.
        n : int
            Size of the default grid. Ignored if `wgrid` is supplied.
            Defaults to 2000.
        wgrid : array
            Integration grid. If not supplied, a default symmetric grid will be
            supplied.
        c_a : tuple
            Extra arguments for `c`
        c_kw : dict
            Extra keyword arguments for `c`

    Notes
    -----
    Uses (1+coth(\omega/2/kt)*c(\omega)/\omega^2*(e^(-i\omega t)+1\omega t -1).
    The original function 'c' is stored as 'g.c'.
    The integrand function is stored as 'g.integrand'

    Equations used herein can be found in
    - Mukamel, Principles of Non-linear optical spectroscopy, Chapter 8
    - V. Butkus, L. Valkunas, and D. Abramavicius, J. Chem. Phys. 137, 8231
      (2012).
    """
    def integrand(t, w, kt, c_a=[], c_kw={}):
        wt = w*t
        ingr = (
           (1 + coth(w/2/kt))
           *c(w,*c_a,**c_kw)/w**2
           *(np.exp(-1j*wt) + 1j * wt - 1)
        )
        return ingr

    def g(t, kt, lim=20, n=2000, wgrid=None, c_a=[], c_kw={}):
        """
        Lineshape function, computed from the spectral correlation function.

        The correlation function is stored under g.c
        """
        if wgrid is None:
            wgrid = np.linspace(-lim, lim, n)
        wgrid = np.reshape(wgrid, (1, -1))
        assert t.ndim == 1
        t = np.reshape(t, (-1, 1))
        ingr = integrand(t, wgrid, kt, c_a=c_a, c_kw=c_kw)
        return -simps(ingr, wgrid, even='last', axis=1)/twopi
    g.c = c
    g.integrand = integrand
    return g


def lorentz_sd(omega, omega_0, gamma):
    """
    Lorentz function for the odd spectral density \tilde{C}''(\omega).

    Parameters
    ----------
    omega : array-like, floats
        Angular frequencies
    omega_0 : float
        Vibrational oscillation frequency
    gamma : float
        Dissipation gamma

    Notes
    -----
    From V. Butkus, L. Valkunas, and D. Abramavicius, J. Chem. Phys. 137, 8231
    (2012).
    """
    n = omega*gamma
    l = omega**2-omega_0**2-gamma**2
    d = l*l+n*n*4
    return n/d*omega_0**3*4


def mbo_sd(omega, omega_0, gamma):
    """
    Odd spectral density \tilde{C}''(\omega) for the MBO model.

    The MBO is the Markovian Brownian oscillator model, a possible component of
    the multimode brownian oscillator model with \gamma(\omega) = \gamma.

    Parameters
    ----------
    omega : array-like, floats
        Angular frequencies
    omega_0 : float
        Vibrational oscillation frequency
    gamma : float
        Dissipation gamma

    Notes
    -----
    From Mukamel, Principles of Non-linear optical spectroscopy, eq 8.64b
    """
    n = omega*gamma
    l = omega**2-omega_0**2
    d = l*l+n*n
    # not sure where that 2sqrt(2) comes from!!!
    return n/d*omega_0**3*2*np.sqrt(2)


def _abs_integrand(omega, t_grid, omega_eg, gs):
    y = np.exp(1j*(omega-omega_eg)*t_grid[:,np.newaxis])
    for g in gs:
        y *= np.exp(-g(t_grid)[:,np.newaxis])
    return y


def linear_absorption(omega, omega_eg, gs, tmax=100, n=5000, t_grid=None):
    """
    Linear absorption lineshape for a sum of lineshape functions.
    """
    if t_grid is None:
        t_grid = np.linspace(0, tmax, n)
    omega = np.reshape(omega, (1, -1))
    y = _abs_integrand(omega, t_grid, omega_eg, gs)
    return simps(y, x=t_grid, axis=0, even='first').real
