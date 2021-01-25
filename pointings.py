from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy import wcs

from chainconsumer import ChainConsumer

import csv
import numpy as np
from scipy import stats
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import os

import config
import databasefunctions as db
import workerfunctions

plt.style.use(astropy_mpl_style)


# Define constants
DIREC = os.getcwd()
sgrA_dist = 7861.2597*3.086e+18  # Distance to Sgr A* from pc to cm
pointing_size = 19.608  # Beam size in arcsec
pointing_center = [
    SkyCoord('17h45m40.60', '-29d00m20.00'), 
    SkyCoord('17h45m43.80', '-29d00m20.00'),
    SkyCoord('17h45m41.75', '-28d59m47.69'),
    SkyCoord('17h45m39.49', '-28d59m51.02'),
    SkyCoord('17h45m38.00', '-29d00m01.00'),
    SkyCoord('17h45m38.30', '-29d00m40.00'),
    SkyCoord('17h45m39.95', '-29d01m03.50'),
    SkyCoord('17h45m41.32', '-29d01m01.63'),
]

pointing_name = ["N", "G", "D", "H", "J", "K", "L", "M"]

pointing_color = ["deepskyblue", "mediumspringgreen", "firebrick", "gold", "steelblue", "lemonchiffon", "magenta", "cyan"]

beams = []

# Read the image data from Farhad
fits_table_filename = 'data/images/HCN.fits'
observation = fits.open(fits_table_filename)[0]
image_data = observation.data[0]
w = wcs.WCS(observation.header)
w = w.dropaxis(-1) # Drops the 3rd axis (not needed)

# Determine size of image from pixel coords and use to determine pixel sizes
origin_x_pos, origin_y_pos = w.wcs_pix2world(0, 0, 1)
max_x_pos, max_y_pos = w.wcs_pix2world(len(image_data), len(image_data[0]), 1)

pixel_width = sgrA_dist*((origin_x_pos - max_x_pos)/len(image_data))
pixel_height = sgrA_dist*((max_y_pos - origin_y_pos)/len(image_data[0]))

############################## Plot the data ############################

# Initialize figure
fig = plt.figure(figsize=(7, 10))
ax = plt.subplot(projection=w, label="overlays")
plt.grid(color='w', linestyle='-', linewidth=1)

# Plot the data
plot = ax.imshow(image_data, interpolation="nearest",
                 origin="lower", cmap="bone", norm=Normalize(vmin=-2.0, vmax=12.0))
# cbaxes = fig.add_axes([0.35, 0.04, 0.45, 0.02])
# cb = fig.colorbar(plot, orientation="horizontal", cax=cbaxes, label="Flux density [Jy/beam km/s]")
# cb.ax.tick_params(labelsize=14)
ax.set_xlim([1200, 2800])
ax.set_ylim([500, 3500])
ax.set_xlabel("Right Ascension (J2000)", fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.set_ylabel("Declination (J2000)", fontsize=16)
ax.tick_params(axis='y', labelsize=16)

# Add the grid
g = ax.grid(alpha=0.25, color='white')

# Add on the sources
for indx, entry in enumerate(pointing_center):  # data[0] is the header row

    # Determine size of beam in cm
    source_radial_extent = pointing_size
    source_size = sgrA_dist*(pointing_size/3600) # Converts arcsec to deg 
    source_pixels = source_size/pixel_height
    
    # Convert to pixels
    pix_coords = w.all_world2pix(entry.ra, entry.dec, 1)
    
    # This is a hack; add_patch doesn't allow the same patch to be
    # added to multiple axes, so make copies of the patch for each
    # axis and plot
    patch = patches.Circle((pix_coords[0], pix_coords[1]), source_pixels, color=pointing_color[indx], alpha=0.3)
    ax.add_patch(patch)

    # Add the source name
    ax.annotate(pointing_name[indx], xy=(pix_coords[0], pix_coords[1]), color='w', size=16)


############################## Overlay best fit data ############################

# plt.margins(0, 0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('data/images/pointings.pdf')
