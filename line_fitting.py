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

source_names = np.array([entry[0] for entry in data])
source_names = np.char.upper(source_names)

# Define line colour
colour = "g"

# Loop through files
for file in files:
    plot_file_name = file[file.index("src"):-4]
    source_name = plot_file_name[plot_file_name.index("_")+1:].upper()
    if source_name == "3" or source_name == "5":
        continue
    x_vals, flux_density = np.loadtxt(file, skiprows=8, unpack=True)
    flux_density = flux_density*1e3 # Convert to mJy
    
    # Get Marc's fit
    existing_sources_indx = [i for i, x in enumerate(list(source_names)) if x == source_name]

    intensities = []

    for source_indx, source in enumerate(existing_sources_indx):
        vel_lsr = float(data[source][-3])
        linewidth_kms = float(data[source][-2])
        peak_flux_density = float(data[source][-4][:data[source][-4].index("±")])
        species = data[source][4]
        
        print("source={0}".format(source_name))
        print("vel_lsr={0}".format(vel_lsr))
        print("linewidth_kms={0}".format(linewidth_kms))
        print("peak_flux_density={0}".format(peak_flux_density))
        print("species={0}".format(species))
        print("\n")

        if species == "SiO 7–6":
            line_freq = 303.92696000
        if species == "OCS 25–24":
            line_freq = 303.99326100
        if species == "SO 8(7)–7(6)":
            line_freq = 304.07784400
        if species == "CH3OH 2(-1)–2(0)":
            line_freq = 304.20834800
        if species == "H2CS 9(1,9)–8(1,8)":
            line_freq = 304.30774000

        # Check to see whether file is in frequency or velocity space
        if x_vals[0] > 0:
            # print("In frequency space")
            obs_freqs = x_vals
            velocities = ((line_freq/obs_freqs) - 1.0)*3e5
            
        elif x_vals[0] < 0:
            # print("In velocity space")
            velocities = x_vals
            obs_freqs = line_freq/((velocities/3e5) + 1.0)
            
        # Get fraction of max intensity in each bin
        fracs = np.exp(-4.0*np.log(2.0)*(((velocities-vel_lsr)**2.0)/(linewidth_kms**2)))
        # Calculate model y axis
        intensity = peak_flux_density*fracs 
        
        # Essentially merges two spectra to ensure continuous curve
        if source_indx > 0:
            for i in range(0, len(intensity)):
                if intensity[i] > spectra[i]:
                    spectra[i] = intensity[i]
        else:
            spectra = intensity

        plt.annotate(species, xy=(obs_freqs[list(intensity).index(np.max(intensity))], peak_flux_density*1.04), xycoords="data", size=8)

    plt.plot(obs_freqs, spectra, color=colour)
    plt.plot(obs_freqs, flux_density, color="b",alpha=0.3, linewidth=0.4, label="Raw spectra")
    plt.annotate(source_name, xy=(obs_freqs[-1], 0.95*np.max(flux_density)), xycoords="data", size=12)
    plt.xlabel("$\\nu$ (GHz)")
    plt.ylabel("Flux density (mJy/beam)")
    plt.savefig("{0}/data/sio_spectra/spectra/{1}.png".format(DIREC, plot_file_name))
    plt.close()
