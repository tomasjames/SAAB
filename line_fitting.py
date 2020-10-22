import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models
from astropy import units as u

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines

import glob
import os


# Define constants
DIREC = os.getcwd()

# Get list of datafiles
files = glob.glob("{0}/data/sio_spectra/*.txt".format(DIREC))

# Loop through files
for file in files:
    print(file)
    plot_file_name = file[file.index("src"):-4]
    source_name = plot_file_name[plot_file_name.index("_"):]
    nu, flux_density = np.loadtxt(file, skiprows=8, unpack=True)

    plt.figure()
    plt.plot(nu, flux_density)
    plt.xlabel("$\\nu$ (GHz)")
    plt.ylabel("Flux density (mJy/beam)")
    plt.savefig(
        "{0}/data/sio_spectra/spectra/{1}.png".format(DIREC, plot_file_name))
    plt.close()
