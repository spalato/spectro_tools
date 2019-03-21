import numpy as np
from .utils import regrid2d, rescale
import matplotlib.pyplot as plt

def lin_alpha_map(amp, offset=1 / 6, slope=1 / 0.8):
    alpha = rescale(amp) - offset
    alpha *= slope / np.max(alpha)
    print(np.max(alpha), np.min(alpha))
    return np.clip(alpha, 0, 1)


def phase_plot(x, y, z,
               phase_cmap="hsv",
               alpha_map=lin_alpha_map,
               cmap_kw={},
               contour_kw={},
               ):
    cmap_defaults = dict(extent=(np.min(x), np.max(x),
                                 np.min(y), np.max(y)),
                         origin="lower",
                         aspect="auto")
    contour_defaults = dict(linewidths=0.5,
                            colors="k",
                            levels=np.linspace(0, 1, 11),
                            )
    cmap_kw = {**cmap_defaults, **cmap_kw}
    lev_kw = {**contour_defaults, **contour_kw}
    # regrid
    x_g, y_g, z_g = regrid2d(x, y, z)
    phase = 0.5 * (np.angle(z_g) / np.pi + 1)
    amp_g = rescale(np.abs(z_g))
    if phase_cmap is None:
        phase_cmap = plt.cm.hsv
    elif isinstance(phase_cmap, str):
        phase_cmap = plt.get_cmap(phase_cmap)
    # else: assume it's callable
    colors = phase_cmap(phase.T)
    colors[:, :, -1] = alpha_map(amp_g.T)
    phase_cmap = plt.imshow(colors, **cmap_kw)
    contours = plt.contour(x, y, rescale(np.abs(z.T)), **lev_kw)
    return phase_cmap, contours

