from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy import wcs

import csv
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import os

import config
import databasefunctions as db

plt.style.use(astropy_mpl_style)


# Define constants
DIREC = os.getcwd()
sgrA_dist = 7861.2597*3.086e+18  # Distance to Sgr A* from pc to cm
beam_size = 0.37  # Beam size in arcsec
x_pix, y_pix = [], []
mean_T, mean_n, mean_N_SIO, mean_N_SO = [], [], [], []

# Define the datafile
datafile = "{0}/data/tabula-sio_intratio.csv".format(DIREC)

# Read in the whole data file containing sources and source flux
with open(datafile, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    f.close()

# Read in the correct config file from the database
config_file = config.config_file(
    db_init_filename='database.ini', section='radex_fit_results')
db_pool = db.dbpool(config_file)

# Read the image data from Farhad
fits_table_filename = 'data/images/20cm.fits'
observation = fits.open(fits_table_filename)[0]
image_data = observation.data[0][0]
w = wcs.WCS(observation.header)
w = w.dropaxis(-1) # Drops the 4rd axis (not needed)
w = w.dropaxis(-1)  #  Drops the 3rd axis (not needed)

# Determine size of image from pixel coords and use to determine pixel sizes
origin_x_pos, origin_y_pos = w.wcs_pix2world(0, 0, 1)
max_x_pos, max_y_pos = w.wcs_pix2world(len(image_data), len(image_data[0]), 1)

pixel_width = sgrA_dist*((origin_x_pos - max_x_pos)/len(image_data))
pixel_height = sgrA_dist*((max_y_pos - origin_y_pos)/len(image_data[0]))

############################## Plot the data ############################

# Initialize figure
fig, ax = plt.subplots(1, figsize=(9, 10))

# Plot the data
plot = ax.imshow(image_data, interpolation="nearest", origin="lower",
                 cmap="bone", norm=Normalize(vmin=0.0, vmax=0.005))
cbaxes = fig.add_axes([0.15, 0.08, 0.6, 0.02])
fig.colorbar(plot, orientation="horizontal", cax=cbaxes, extend='max', label="Flux density [Jy/beam]")
ax.set_xlim([0, 3500])

# Add the grid
g = ax.grid(alpha=0.25, color='white')

# Zoom in by a factor of 4.0 and place on the upper left
axins = zoomed_inset_axes(ax, 8.0, loc=2, bbox_to_anchor=(35, 650), borderpad=1)
axins.imshow(image_data, interpolation="nearest", origin="lower", 
            cmap="bone", norm=Normalize(vmin=0.0, vmax=0.005))
plt.yticks(visible=False)
plt.xticks(visible=False)

# Zoom in by a factor of 4.0 and place on the lower right
axins1 = zoomed_inset_axes(ax, 6.0, loc=3, bbox_to_anchor=(35, 80), borderpad=1)
axins1.imshow(image_data, interpolation="nearest", origin="lower",
            cmap="bone", norm=Normalize(vmin=0.0, vmax=0.005))
plt.yticks(visible=False)
plt.xticks(visible=False)

beams, beams_inset1, beams_inset2 = [], [], []

# Add on the sources
for indx, entry in enumerate(data[1:]):  # data[0] is the header row

    # Determine size of beam in cm
    source_radial_extent = float(entry[3])*beam_size
    source_size = sgrA_dist*(beam_size/3600) # Converts arcsec to deg 
    source_pixels = source_size/pixel_height

    # Read in the data for the given source
    best_fit_data = db.get_bestfit(
        db_pool=db_pool,
        table=entry[0],
        column_names=[
            "temp",
            "dens",
            "column_density_sio",
            "column_density_so"
        ]
    )
    best_fit_data = np.array(best_fit_data) # Converts to array 

    # Only need to pull out the coordinates of each source 
    coords = SkyCoord(
        ra=Angle('17h45m{0}s'.format(entry[1])), dec=Angle('-29d00m{0}s'.format(entry[2])), frame='fk4')
    
    # Convert to pixels
    pix_coords = w.all_world2pix(coords.ra.deg, coords.dec.deg, 1)

    # Store the mean temperature and pixel info
    # for sources of interest
    if best_fit_data != []:
        mean_T.append(np.mean(best_fit_data[:, 0]))
        mean_n.append(np.mean(best_fit_data[:, 1]))
        mean_N_SIO.append(np.mean(best_fit_data[:, 2]))
        mean_N_SO.append(np.mean(best_fit_data[:, 3]))
        x_pix.append(pix_coords[0]), y_pix.append(pix_coords[1])

    # This is a hack; add_patch doesn't allow the same patch to be
    # added to multiple axes, so make copies of the patch for each
    # axis and plot
    beams.append(patches.Circle((pix_coords[0], pix_coords[1]), source_pixels))
    beams_inset1.append(patches.Circle((pix_coords[0], pix_coords[1]), source_pixels))
    beams_inset2.append(patches.Circle((pix_coords[0], pix_coords[1]), source_pixels))

    # Add the source name
    axins.annotate(entry[0], xy=(pix_coords[0], pix_coords[1]), color='w', size=5)
    axins1.annotate(entry[0], xy=(pix_coords[0], pix_coords[1]), color='w', size=5)

############################## Overlay best fit data ############################

# Define the collections to house all of the patches
p = PatchCollection(beams, cmap="rainbow")
p_inset1 = PatchCollection(beams_inset1, cmap="rainbow")
p_inset2 = PatchCollection(beams_inset2, cmap="rainbow")

# Plot these collections on the correct axis
ax.add_collection(p)
axins.add_collection(p_inset1)
axins1.add_collection(p_inset2)

# Sets the ranges on the collections to their correct min and max values
<<<<<<< HEAD
p.set_array(np.array(mean_n))
p.set_clim([np.ma.min(mean_n), np.ma.max(mean_n)])

p_inset1.set_array(np.array(mean_n))
p_inset1.set_clim([np.ma.min(mean_n), np.ma.max(mean_n)])

p_inset2.set_array(np.array(mean_n))
p_inset2.set_clim([np.ma.min(mean_n), np.ma.max(mean_n)])
=======
p.set_array(np.array(mean_N_SO))
p.set_clim([np.ma.min(mean_N_SO), np.ma.max(mean_N_SO)])

p_inset1.set_array(np.array(mean_N_SO))
p_inset1.set_clim([np.ma.min(mean_N_SO), np.ma.max(mean_N_SO)])

p_inset2.set_array(np.array(mean_N_SO))
p_inset2.set_clim([np.ma.min(mean_N_SO), np.ma.max(mean_N_SO)])
>>>>>>> 72c633029c2759fbe561733dd1cc02fc81b03635

# Adds the colour bar
plt.colorbar(p, ax=ax, label="N$_{SO, mean}$ [cm$^{-2}$]")

# Change the grid on the inset
axins.grid(alpha=0.1, color='white')
axins1.grid(alpha=0.1, color='white')

x1, x2, y1, y2 = 2000, 2140, 2250, 2410  # specify the limits
axins.set_xlim(x1, x2)  # apply the x-limits
axins.set_ylim(y1, y2)  # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")

x1, x2, y1, y2 = 2225, 2424, 1640, 2025  # specify the limits
axins1.set_xlim(x1, x2)  # apply the x-limits
axins1.set_ylim(y1, y2)  # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins1, loc1=4, loc2=2, fc="none", ec="0.5")

# plt.tight_layout()
<<<<<<< HEAD
fig.savefig('data/images/20cm_n_mean.pdf')
=======
fig.savefig('data/images/20cm_N_SO_mean.pdf')
>>>>>>> 72c633029c2759fbe561733dd1cc02fc81b03635
