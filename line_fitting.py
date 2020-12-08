import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models
from astropy import units as u

from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import fit_lines, find_lines_threshold, estimate_line_parameters
from specutils.manipulation import extract_region, noise_region_uncertainty

import csv
import glob
import os


# Define constants
DIREC = os.getcwd()

# Get list of datafiles
files = glob.glob("{0}/data/sio_spectra/*.txt".format(DIREC))

# Load Marc's data
with open("{0}/data/tabula-sio_intratio.csv".format(DIREC), newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    f.close()

# Remove header row
data.remove(data[0])

source_names = np.array(data)[1:, 0]

# Define line colour
colour = "g"

# Loop through files
for file in files:
    plot_file_name = file[file.index("src"):-4]
    source_name = plot_file_name[plot_file_name.index("_")+1:].upper()
    # if source_name != "C1":
    #     continue
    nu, flux_density = np.loadtxt(file, skiprows=8, unpack=True)
    flux_density = flux_density*1e3 # Convert to mJy

    model_nu = np.linspace(nu[0], nu[-1], len(nu))

    # Get Marc's fit
    existing_sources_indx = [i for i, x in enumerate(list(source_names)) if x == source_name]

    plt.figure()

    for source_indx, source in enumerate(existing_sources_indx):
        velocity = float(data[source][-3])
        linewidth_kms = float(data[source][-2])
        peak_flux_density = float(data[source][-4][:data[source][-4].index("±")])
        line_freq = nu[min(range(len(flux_density)), key=lambda i: abs(flux_density[i]-peak_flux_density))]

        print(line_freq)

        # Convert linewidth to frequency space
        linewidth = abs(line_freq * (linewidth_kms/3e5))

        # Fit Gaussian to data
        lower_bound = (np.abs(nu - (line_freq - linewidth))).argmin()
        upper_bound = (np.abs(nu - (line_freq + linewidth))).argmin()
        m = models.Gaussian1D(amplitude=peak_flux_density, mean=line_freq, stddev=linewidth)

        # Generate fit array for plotting
        x = np.linspace(nu[upper_bound]+0.04, nu[lower_bound]-0.04, 101)
        fit = m(x)

        if source_indx == 0:
            domain = [model_nu[-1], model_nu[lower_bound]-0.04]
            prev_line_upper_bound = nu[upper_bound]+0.04
            print(upper_bound)
        elif source_indx == len(existing_sources_indx) - 1:
            plt.plot([model_nu[upper_bound]+0.04, model_nu[0]], [0, 0], color=colour)
            domain = [prev_line_upper_bound, nu[lower_bound]-0.04]
        else:
            domain = [prev_line_upper_bound, nu[lower_bound]-0.04]
            prev_line_upper_bound = nu[upper_bound]+0.04
        print(domain)
        
        plt.plot(domain, [0, 0], color=colour)

        plt.plot(x, fit, color=colour)

    # plt.plot(model_nu, np.linspace(0, 0, len(model_nu)), color=colour)
    plt.plot(nu, flux_density, color="b", alpha=0.5, linewidth=0.4, label="Raw spectra")
    plt.xlabel("$\\nu$ (GHz)")
    plt.ylabel("Flux density (mJy/beam)")
    # plt.annotate(source_name, (np.max(nu)*0.9997, np.max(flux_density)*0.98))
    plt.savefig(
        "{0}/data/sio_spectra/spectra/{1}.png".format(DIREC, plot_file_name))
    plt.close()
