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
config_file = config.config_file(db_init_filename='database_archi.ini', section='radex_fit_results')
bestfit_config_file = config.config_file(db_init_filename='database_archi.ini', section='radex_bestfit_conditions')

db_pool = db.dbpool(config_file)
db_bestfit_pool = db.dbpool(bestfit_config_file)

# Parse the data to a dict list
observed_data = workerfunctions.parse_data(
    data, db_pool, db_bestfit_pool)

# Filter the observed data to contain only those species that we can use
# (normally limited by those with Radex data)
filtered_data = workerfunctions.filter_data(
    observed_data, ["SIO", "SO", "OCS", "H2CS"])

# Read the image data from Farhad
fits_table_filename = 'data/images/plw.fits'
observation = fits.open(fits_table_filename)[1]
image_data = observation.data
w = wcs.WCS(observation.header)

# Determine size of image from pixel coords and use to determine pixel sizes
origin_x_pos, origin_y_pos = w.wcs_pix2world(0, 0, 1)
max_x_pos, max_y_pos = w.wcs_pix2world(len(image_data), len(image_data[0]), 1)

pixel_width = sgrA_dist*((origin_x_pos - max_x_pos)/len(image_data))
pixel_height = sgrA_dist*((max_y_pos - origin_y_pos)/len(image_data[0]))

############################## Plot the data ############################
# Initialize figure
fig, ax = plt.subplots(1, figsize=(9, 12))
# fig.subplots_adjust(left=0.1, bottom=0.1)

# Plot the data
plot = ax.imshow(np.log10(image_data), interpolation="nearest", origin="lower", cmap="bone")
# cbaxes = fig.add_axes([0.2, 0.04, 0.6, 0.02])
fig.colorbar(plot, orientation="horizontal", label="Flux density [Jy/beam km/s]")
# ax.set_xlim([1200, 2800])
# ax.set_ylim([500, 3500])
# ax.set_xlabel("Right Ascension (J2000)")
# ax.set_ylabel("Declination (J2000)")

'''
# Add the grid
g = ax.grid(alpha=0.25, color='white')

# Zoom in by a factor of 4.0 and place on the upper left
axins = zoomed_inset_axes(ax, 6.0, loc=2, bbox_to_anchor=(40, 800), borderpad=1)
axins.imshow(image_data, interpolation="nearest", origin="lower",
             cmap="bone", norm=Normalize(vmin=-2.0, vmax=8.0))
plt.yticks(visible=False)
plt.xticks(visible=False)

# Zoom in by a factor of 4.0 and place on the lower right
axins1 = zoomed_inset_axes(ax, 4.0, loc=3, bbox_to_anchor=(40, 50), borderpad=1)
axins1.imshow(image_data, interpolation="nearest", origin="lower",
              cmap="bone", norm=Normalize(vmin=-2.0, vmax=8.0))
plt.yticks(visible=False)
plt.xticks(visible=False)

# axins2 = zoomed_inset_axes(ax, 4.0, loc=2, bbox_to_anchor=(40, 60), borderpad=1)
# axins2.imshow(image_data, interpolation="nearest", origin="lower",
#               cmap="bone", norm=Normalize(vmin=-2.0, vmax=8.0))
# plt.yticks(visible=False)
# plt.xticks(visible=False)

beams, beams_inset1, beams_inset2, beams_inset3 = [], [], [], []

# Add on the sources
for indx, entry in enumerate(data[1:]):  # data[0] is the header row

    # Determine size of beam in cm
    source_radial_extent = float(entry[3])*beam_size
    source_size = sgrA_dist*(beam_size/3600) # Converts arcsec to deg 
    source_pixels = source_size/pixel_height

    # Set up chain
    c = ChainConsumer()

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

    # best_fit_data = np.array(best_fit_data) # Converts to array 
    # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
    if best_fit_data is None or best_fit_data == []:
        continue
    chain = np.array(best_fit_data[int(len(best_fit_data)*0.2):])
    print("chain={0}".format(chain))
    c.add_chain(chain, parameters=[
            "temp",
            "dens",
            "column_density_sio",
            "column_density_so"
        ], 
        walkers=200, name=entry[0])

    # Get summary
    summary = c.analysis.get_summary()

    # Only need to pull out the coordinates of each source 
    coords = SkyCoord(
        ra=Angle('17h45m{0}s'.format(entry[1])), dec=Angle('-29d00m{0}s'.format(entry[2])), frame='fk4')
    
    # Convert to pixels
    pix_coords = w.all_world2pix(coords.ra.deg, coords.dec.deg, 1)

    # Store the mean temperature and pixel info
    # for sources of interest
    if best_fit_data != []:
        if summary["temp"][0] == None:
            continue
        mean_T.append(summary["temp"][1])
        if summary["dens"][0] == None:
            continue
        mean_n.append(summary["dens"][1])
        if summary["column_density_sio"][0] == None:
            continue
        mean_N_SIO.append(summary["column_density_sio"][1])
        if summary["column_density_so"][0] == None:
            continue
        mean_N_SO.append(summary["column_density_so"][1])

        x_pix.append(pix_coords[0]), y_pix.append(pix_coords[1])
    
    # This is a hack; add_patch doesn't allow the same patch to be
    # added to multiple axes, so make copies of the patch for each
    # axis and plot
    beams.append(patches.Circle((pix_coords[0], pix_coords[1]), source_pixels))
    beams_inset1.append(patches.Circle((pix_coords[0], pix_coords[1]), source_pixels))
    beams_inset2.append(patches.Circle((pix_coords[0], pix_coords[1]), source_pixels))
    #beams_inset3.append(patches.Circle((pix_coords[0], pix_coords[1]), source_pixels))

    # Add the source name
    axins.annotate(entry[0], xy=(pix_coords[0], pix_coords[1]), color='w', size=5)
    axins1.annotate(entry[0], xy=(pix_coords[0], pix_coords[1]), color='w', size=5)
    
    #if entry[0][0] == "G":
    #    ax.annotate(entry[0], xy=(pix_coords[0], pix_coords[1]), color='w', size=5)

############################## Overlay best fit data ############################

# Define the collections to house all of the patches
p = PatchCollection(beams, cmap="rainbow")
p_inset1 = PatchCollection(beams_inset1, cmap="rainbow")
p_inset2 = PatchCollection(beams_inset2, cmap="rainbow")
#p_inset3 = PatchCollection(beams_inset3, cmap="rainbow")

# Plot these collections on the correct axis
ax.add_collection(p)
axins.add_collection(p_inset1)
axins1.add_collection(p_inset2)
#axins2.add_collection(p_inset3)

# Sets the ranges on the collections to their correct min and max values
p.set_array(np.array(mean_T))
p.set_clim([np.ma.min(mean_T), np.ma.max(mean_T)])

p_inset1.set_array(np.array(mean_T))
p_inset1.set_clim([np.ma.min(mean_T), np.ma.max(mean_T)])

p_inset2.set_array(np.array(mean_T))
p_inset2.set_clim([np.ma.min(mean_T), np.ma.max(mean_T)])

#p_inset3.set_array(np.array(mean_n))
#p_inset3.set_clim([np.ma.min(mean_n), np.ma.max(mean_n)])

# Adds the colour bar
#plt.colorbar(p, ax=ax, label="N$_{SO, mean}$ [cm$^{-2}$]")
#plt.colorbar(p, ax=ax, label="n$_{mean}$ [cm$^{-3}$]")
plt.colorbar(p, ax=ax, label="T$_{mean}$ [K]")

# Change the grid on the inset
axins.grid(alpha=0.1, color='white')
axins1.grid(alpha=0.1, color='white')

x1, x2, y1, y2 = 1350, 1750, 2000, 2400  # specify the limits
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

x1, x2, y1, y2 = 1400, 1800, 2000, 4300  # specify the limits
axins2.set_xlim(x1, x2)  # apply the x-limits
axins2.set_ylim(y1, y2)  # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins2, loc1=4, loc2=3, fc="none", ec="0.5")
'''
# plt.tight_layout()
fig.savefig('data/images/dust_plw.pdf')
