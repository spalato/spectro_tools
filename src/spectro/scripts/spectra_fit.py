# usage: python spectra.py {oceanview_file.txt}
# usage: ls *.txt | %{python .\spectra.py $_.Name}

import numpy as np
from lmfit.models import ConstantModel, GaussianModel
from spectro.utils import brent_fwhm
from spectro.io import read_oceanview
import matplotlib.pyplot as plt
import argparse
import logging
import os.path as pth
import typer
from typing_extensions import Annotated


logging.basicConfig()
logger = logging.getLogger()
logging.getLogger("matplotlib").setLevel(logging.ERROR)

app = typer.Typer()

@app.command()
def spectra_fit(
        filename: str,
        verbose: bool = False,
        outroot: Annotated[
            str, typer.Option(help="Root for saved file names.")
        ] = None,
        fit_report: Annotated[
            bool, typer.Option(help="Save fit report")
        ] = False,
):

    """
    Analyze spectrum in `filename`, measuring central wavelength and FWHM.

    "Processes spectra to measure bandwidth and central wavelength. Outputs to a graph and (optionnaly) to a file.

    Powershell usage: `ls *.txt | %{spectra_fit $_.Name}`
    """
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    outroot = pth.splitext(filename)[0] if outroot is None else outroot
    logger.debug("outroot is :"+outroot)

    logger.info("Loading file: %s", filename)
    wl, y = read_oceanview(filename)
    # clip the first pixels, which have a different baseline.
    wl = wl[40:]
    y = y[40:]

    # Fitting a simple gaussian model
    logging.info("Performing gaussian fit.")
    logging.debug("Setting up model and guess.")
    model = ConstantModel() + GaussianModel()

    bl = np.median(y)
    sig = 20  # 20 nm
    amp = np.max(y - bl) * sig * np.sqrt(2)
    cwl = wl[np.argmax(y)]

    guess = model.make_params()
    guess["c"].set(value=bl)
    guess["amplitude"].set(value=amp, min=0)
    guess["sigma"].set(value=sig, min=cwl / 10000, max=cwl * 4)
    guess["center"].set(value=cwl, min=np.min(wl), max=np.max(wl))

    logging.debug("Performing fit.")
    result = model.fit(y, guess, x=wl)
    bl_fit = result.best_values["c"]
    cwl_fit = result.best_values["center"]
    sig_fit = result.best_values["sigma"]
    fwhm_fit = result.params["fwhm"].value

    if fit_report:
        fit_report_fn = outroot + "_fitreport.txt"
        logging.info("Saving fit report to: " + fit_report_fn)
        with open(fit_report_fn, 'w') as f:
            f.write(result.fit_report())

    # Numerical analysis:
    logging.info("Performing numerical width analysis.")
    margin = 6 * sig_fit

    # will not work for very broad pulses, if there is no data left outside of the margins.
    m = (wl > cwl_fit - margin) & (wl < cwl_fit + margin)
    bl = np.mean(y[~m])
    cwl_num = np.average(wl[m], weights=(y - bl)[m])
    fwhm_num, (x1, x2) = brent_fwhm(wl, y - bl, shift=False)

    logging.debug("Preparing report")
    report = f"""filename: {filename}
      barycenter: {cwl_num:0.06g}
      fwhm_num:   {fwhm_num:0.06g}
      cwl_fit:    {cwl_fit:0.06g}
      fwhm_fit:   {fwhm_fit:0.06g}
    """
    spectra_report_fn = outroot + "_spectra.txt"
    logging.info("Saving report to: " + spectra_report_fn)
    with open(spectra_report_fn, "w") as f:
        f.write(report)

    logging.debug("Preparing figure")
    plt.figure()
    plt.plot(wl, y, "k-", label="Data")
    plt.plot(wl, result.best_fit, "r-", lw=0.8, label="Gaussian fit")
    plt.axvline(cwl_num, color="b", lw=0.8, label=f"Barycenter")

    h = np.max(y - bl) / 2 + bl
    plt.plot([x1, x2], [h, h], "o-b", label="FWHM num")
    plt.axvline(cwl_fit, color="r", ls=":", label=r"$\lambda_0$ fit")

    plt.legend()
    plt.text(0.05, 0.95, "\n".join(report.splitlines()[1:]),
             bbox=dict(
                 boxstyle="square",
                 alpha=0.5,
                 fc=(1.0, 1.0, 1.0),
                 ec="none",
             ),
             va="top",
             ha="left",
             fontweight="light",
             fontsize="small",
             transform=plt.gca().transAxes,
             )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectra (counts)")
    plt.suptitle(report.splitlines()[0])
    plt.xlim(cwl_fit - margin, cwl_fit + margin)
    plt.tight_layout()
    fig_fn = outroot + ".png"
    logging.info("Saving figure to: " + fig_fn)
    plt.savefig(fig_fn, dpi=300)
