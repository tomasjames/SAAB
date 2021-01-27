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


def plotting(data):
    p.set_array(np.array(data))

    p.set_clim([np.ma.min(data), np.ma.max(data)])

    p_inset1.set_array(np.array(data))
    p_inset1.set_clim([np.ma.min(data), np.ma.max(data)])

    p_inset2.set_array(np.array(data))
    p_inset2.set_clim([np.ma.min(data), np.ma.max(data)])

    p_inset3.set_array(np.array(data))
    p_inset3.set_clim([np.ma.min(data), np.ma.max(data)])

    # Adds the colour bar
    plt.colorbar(
        p, ax=ax, label=r"$\log \left( \mathrm{I}_{\mathrm{SiO}}/\mathrm{I}_{\mathrm{SO}} \right)$")

    # Change the grid on the inset
    axins.grid(alpha=0.1, color='white')
    axins1.grid(alpha=0.1, color='white')
    axins2.grid(alpha=0.1, color='white')

    x1, x2, y1, y2 = 1400, 1700, 2000, 2320  # specify the limits
    axins.set_xlim(x1, x2)  # apply the x-limits
    axins.set_ylim(y1, y2)  # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")

    x1, x2, y1, y2 = 1995, 2135, 1510, 1690  # specify the limits
    axins1.set_xlim(x1, x2)  # apply the x-limits
    axins1.set_ylim(y1, y2)  # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins1, loc1=1, loc2=4, fc="none", ec="0.5")

    x1, x2, y1, y2 = 2230, 2430, 1750, 2040  # specify the limits
    axins2.set_xlim(x1, x2)  # apply the x-limits
    axins2.set_ylim(y1, y2)  # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.5")

    # plt.tight_layout()
    fig.savefig('data/images/sio-so-ratio.pdf')


# Define constants
DIREC = os.getcwd()
sgrA_dist = 7861.2597*3.086e+18  # Distance to Sgr A* from pc to cm
beam_size = 0.37  # Beam size in arcsec
x_pix, y_pix = [], []

# Define the datafile
datafile = "{0}/data/tabula-sio_intratio.csv".format(DIREC)

# Read in the whole data file containing sources and source flux
with open(datafile, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    f.close()

# Read the image data from Farhad
fits_table_filename = 'data/images/HCN.fits'
observation = fits.open(fits_table_filename)[0]
image_data = observation.data[0]
w = wcs.WCS(observation.header)
w = w.dropaxis(-1)  # Drops the 3rd axis (not needed)

# Determine size of image from pixel coords and use to determine pixel sizes
origin_x_pos, origin_y_pos = w.wcs_pix2world(0, 0, 1)
max_x_pos, max_y_pos = w.wcs_pix2world(len(image_data), len(image_data[0]), 1)

pixel_width = sgrA_dist*((origin_x_pos - max_x_pos)/len(image_data))
pixel_height = sgrA_dist*((max_y_pos - origin_y_pos)/len(image_data[0]))

############################## Plot the data ############################
# Initialize figure
fig, (ax1, ax) = plt.subplots(2, 2, figsize=(18, 17))
ax = plt.subplot(projection=w, label="overlays")
plt.grid(color='w', linestyle='-', linewidth=1)

# Plot the data
plot = ax.imshow(image_data, interpolation="nearest",
                 origin="lower", cmap="bone", norm=Normalize(vmin=-2.0, vmax=12.0))
cbaxes = fig.add_axes([0.35, 0.04, 0.45, 0.02])
cb = fig.colorbar(plot, orientation="horizontal", cax=cbaxes,
                  label="Flux density [Jy/beam km/s]")
cb.ax.tick_params(labelsize=14)
ax.set_xlim([1200, 2800])
ax.set_ylim([500, 3500])
ax.set_xlabel("Right Ascension (J2000)", fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.set_ylabel("Declination (J2000)", fontsize=16)
ax.tick_params(axis='y', labelsize=16)

# Add the grid
g = ax.grid(alpha=0.25, color='white')

# Zoom in by a factor of 4.0 and place on the upper left
axins = zoomed_inset_axes(
    ax, 4.0, loc=2, bbox_to_anchor=(5, 1220), borderpad=1)
axins.imshow(image_data, interpolation="nearest", origin="lower",
             cmap="bone", norm=Normalize(vmin=-2.0, vmax=12.0))
plt.yticks(visible=False)
plt.xticks(visible=False)

# Zoom in by a factor of 4.0 and place on the lower right
axins1 = zoomed_inset_axes(
    ax, 8.0, loc=6, bbox_to_anchor=(5, 400), borderpad=1)
axins1.imshow(image_data, interpolation="nearest", origin="lower",
              cmap="bone", norm=Normalize(vmin=-2.0, vmax=12.0))
plt.yticks(visible=False)
plt.xticks(visible=False)

axins2 = zoomed_inset_axes(
    ax, 5.7, loc=3, bbox_to_anchor=(660, 680), borderpad=1)
axins2.imshow(image_data, interpolation="nearest", origin="lower",
              cmap="bone", norm=Normalize(vmin=-2.0, vmax=12.0))
plt.yticks(visible=False)
plt.xticks(visible=False)

sio_so_ratio = []
beams, beams_inset1, beams_inset2, beams_inset3 = [], [], [], []

# Add on the sources
for indx, entry in enumerate(data[1:]):  # data[0] is the header row

    # Determine size of beam in cm
    source_radial_extent = float(entry[3])*beam_size
    source_size = sgrA_dist*(beam_size/3600)  # Converts arcsec to deg
    source_pixels = source_size/pixel_height

    # Only need to pull out the coordinates of each source
    coords = SkyCoord(
        ra=Angle('{0}h{1}m{2}s'.format(entry[1][0:2], entry[1][3:5], entry[1][6:-1])), dec=Angle('-{0}d{1}m{02}s'.format(entry[2][1:3], entry[2][4:6], entry[2][7:-2])), frame='fk4')

    # Convert to pixels
    pix_coords = w.all_world2pix(coords.ra.deg, coords.dec.deg, 1)

    # Store the ratio and pixel info
    # for sources of interest
    if entry[-1]:
        try:
            ratio = np.log10(abs(float(entry[-1][:entry[-1].find("±")])))
            sio_so_ratio.append(ratio)
        except RuntimeWarning:
            print(entry[-1])
            break

    # This is a hack; add_patch doesn't allow the same patch to be
    # added to multiple axes, so make copies of the patch for each
    # axis and plot
    beams.append(patches.Circle(
        (pix_coords[0], pix_coords[1]), source_pixels))
    beams_inset1.append(patches.Circle(
        (pix_coords[0], pix_coords[1]), source_pixels))
    beams_inset2.append(patches.Circle(
        (pix_coords[0], pix_coords[1]), source_pixels))
    beams_inset3.append(patches.Circle(
        (pix_coords[0], pix_coords[1]), source_pixels))

    # Add the source name
    axins.annotate(entry[0], xy=(
        pix_coords[0], pix_coords[1]), color='w', size=16)
    axins1.annotate(entry[0], xy=(
        pix_coords[0], pix_coords[1]), color='w', size=16)
    axins2.annotate(entry[0], xy=(
        pix_coords[0], pix_coords[1]), color='w', size=16)

    print(entry[0][0])

############################## Overlay best fit data ############################

# Define the collections to house all of the patches
p = PatchCollection(beams, cmap="rainbow")
p_inset1 = PatchCollection(beams_inset1, cmap="rainbow")
p_inset2 = PatchCollection(beams_inset2, cmap="rainbow")
p_inset3 = PatchCollection(beams_inset3, cmap="rainbow")

# Plot these collections on the correct axis
ax.add_collection(p)
axins.add_collection(p_inset1)
axins1.add_collection(p_inset2)
axins2.add_collection(p_inset3)

#Annotate the inset plots
axins.annotate("Panel 1", (1410, 2300), color="w", size=20)
axins2.annotate("Panel 2", (2240, 2020), color="w", size=20)
axins1.annotate("Panel 3", (2005, 1680), color="w", size=20)

# Sets the ranges on the collections to their correct min and max values
plotting(sio_so_ratio)
