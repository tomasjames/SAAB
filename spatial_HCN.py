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


def plotting(param, data, data_trans):
    p.set_array(np.array(data))

    p.set_clim([np.ma.min(data), np.ma.max(data)])

    p_inset1.set_array(np.array(data))
    p_inset1.set_clim([np.ma.min(data), np.ma.max(data)])

    p_inset2.set_array(np.array(data))
    p_inset2.set_clim([np.ma.min(data), np.ma.max(data)])

    p_inset3.set_array(np.array(data))
    p_inset3.set_clim([np.ma.min(data), np.ma.max(data)])

    if data_trans:
        # Sets the ranges on the collections to their correct min and max values
        p_trans.set_array(np.array(data_trans))
        p_trans.set_clim([np.ma.min(data_trans), np.ma.max(data_trans)])

        p_inset1_trans.set_array(np.array(data_trans))
        p_inset1_trans.set_clim([np.ma.min(data_trans), np.ma.max(data_trans)])

        p_inset2_trans.set_array(np.array(data_trans))
        p_inset2_trans.set_clim([np.ma.min(data_trans), np.ma.max(data_trans)])

        p_inset3_trans.set_array(np.array(data_trans))
        p_inset3_trans.set_clim([np.ma.min(data_trans), np.ma.max(data_trans)])

    # Adds the colour bar
    #plt.colorbar(p, ax=ax, label="N$_\mathrm{SO, MLE}$ [cm$^{-2}$]")
    #plt.colorbar(p, ax=ax, label="n$_\mathrm{MLE}$ [cm$^{-3}$]")
    if param == "T":
        plt.colorbar(p, ax=ax, label="T$_\mathrm{MLE}$ [K]")
    if param == "n":
        plt.colorbar(p, ax=ax, label="n$_\mathrm{MLE}$ [cm$^{-3}$]")
    if param == "N_SIO":
        plt.colorbar(p, ax=ax, label="N$_\mathrm{SiO, MLE}$ [cm$^{-2}$]")
    if param == "N_SO":
        plt.colorbar(p, ax=ax, label="N$_\mathrm{SO, MLE}$ [cm$^{-2}$]")

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

    x1, x2, y1, y2 = 2225, 2424, 1650, 2020  # specify the limits
    axins1.set_xlim(x1, x2)  # apply the x-limits
    axins1.set_ylim(y1, y2)  # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins1, loc1=4, loc2=1, fc="none", ec="0.5")

    x1, x2, y1, y2 = 2000, 2130, 2250, 2420  # specify the limits
    axins2.set_xlim(x1, x2)  # apply the x-limits
    axins2.set_ylim(y1, y2)  # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.5")

    # plt.tight_layout()
    print(param)
    fig.savefig('data/images/HCN_{0}_mean.pdf'.format(str(param)))


# Define constants
DIREC = os.getcwd()
sgrA_dist = 7861.2597*3.086e+18  # Distance to Sgr A* from pc to cm
beam_size = 0.37  # Beam size in arcsec
x_pix, y_pix = [], []
mean_T, mean_n, mean_N_SIO, mean_N_SO = [], [], [], []
mean_T_trans, mean_n_trans, mean_N_SIO_trans, mean_N_SO_trans = [], [], [], []
patch_trans = 0.3   
param = "N_SIO"
hatch = "///"

# Define the datafile
datafile = "{0}/data/tabula-sio_intratio.csv".format(DIREC)

# Read in the whole data file containing sources and source flux
with open(datafile, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    f.close()

# Read in the correct config file from the database
config_file = config.config_file(
    db_init_filename='database_archi.ini', section='radex_fit_results')
bestfit_config_file = config.config_file(
    db_init_filename='database_archi.ini', section='radex_bestfit_conditions')

db_pool = db.dbpool(config_file)
db_bestfit_pool = db.dbpool(bestfit_config_file)

# Parse the data to a dict list
observed_data = workerfunctions.parse_data(
    data, db_pool, db_bestfit_pool)

# Filter the observed data to contain only those species that we can use
# (normally limited by those with Radex data)
filtered_data = workerfunctions.filter_data(
    observed_data, ["SIO", "SO", "OCS", "H2CS", "CH3OH"])

jwytfefey

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
axins1 = zoomed_inset_axes(ax, 6.0, loc=3, bbox_to_anchor=(5, 10), borderpad=1)
axins1.imshow(image_data, interpolation="nearest", origin="lower",
              cmap="bone", norm=Normalize(vmin=-2.0, vmax=12.0))
plt.yticks(visible=False)
plt.xticks(visible=False)

axins2 = zoomed_inset_axes(
    ax, 7.0, loc=6, bbox_to_anchor=(600, 1015), borderpad=1)
axins2.imshow(image_data, interpolation="nearest", origin="lower",
              cmap="bone", norm=Normalize(vmin=-2.0, vmax=12.0))
plt.yticks(visible=False)
plt.xticks(visible=False)

beams, beams_inset1, beams_inset2, beams_inset3 = [], [], [], []
beams_trans, beams_inset1_trans, beams_inset2_trans, beams_inset3_trans = [], [], [], []

# Add on the sources
for indx, entry in enumerate(data[1:]):  # data[0] is the header row

    trans = {
        "temp": False,
        "dens": False,
        "SIO": False,
        "SO": False,
    }

    # Determine size of beam in cm
    source_radial_extent = float(entry[3])*beam_size
    source_size = sgrA_dist*(beam_size/3600)  # Converts arcsec to deg
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

    if entry[0] == "G2":
        best_fit_data = best_fit_data[:int(235000)]

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
        walkers=500, name=entry[0])

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
            mean_T_trans.append(summary["temp"][1])
            trans["temp"] = True
        else:
            mean_T.append(summary["temp"][1])
        if summary["dens"][0] == None:
            mean_n_trans.append(summary["dens"][1])
            trans["dens"] = True
        else:
            mean_n.append(summary["dens"][1])
        if summary["column_density_sio"][0] == None:
            mean_N_SIO_trans.append(summary["column_density_sio"][1])
            trans["SIO"] = True
        else:
            mean_N_SIO.append(summary["column_density_sio"][1])
        if summary["column_density_so"][0] == None:
            mean_N_SO_trans.append(summary["column_density_so"][1])
            trans["SO"] = True
        else:
            mean_N_SO.append(summary["column_density_so"][1])

        x_pix.append(pix_coords[0]), y_pix.append(pix_coords[1])

    # This is a hack; add_patch doesn't allow the same patch to be
    # added to multiple axes, so make copies of the patch for each
    # axis and plot
    if trans["temp"] == True and param == "T" or trans["dens"] == True and param == "n" or trans["SIO"] == True and param == "N_SIO" or trans["SO"] == True and param == "N_SO":
        beams_trans.append(patches.Circle(
            (pix_coords[0], pix_coords[1]), source_pixels))
        beams_inset1_trans.append(patches.Circle(
            (pix_coords[0], pix_coords[1]), source_pixels))
        beams_inset2_trans.append(patches.Circle(
            (pix_coords[0], pix_coords[1]), source_pixels))
        beams_inset3_trans.append(patches.Circle(
            (pix_coords[0], pix_coords[1]), source_pixels))
    else:
        beams.append(patches.Circle((pix_coords[0], pix_coords[1]), source_pixels))
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

p_trans = PatchCollection(beams_trans, cmap="rainbow", alpha=patch_trans, hatch=hatch)
p_inset1_trans = PatchCollection(beams_inset1_trans, cmap="rainbow", alpha=patch_trans, hatch=hatch)
p_inset2_trans = PatchCollection(beams_inset2_trans, cmap="rainbow", alpha=patch_trans, hatch=hatch)
p_inset3_trans = PatchCollection(beams_inset3_trans, cmap="rainbow", alpha=patch_trans, hatch=hatch)

# Plot these collections on the correct axis
ax.add_collection(p)
axins.add_collection(p_inset1)
axins1.add_collection(p_inset2)
axins2.add_collection(p_inset3)

ax.add_collection(p_trans)
axins.add_collection(p_inset1_trans)
axins1.add_collection(p_inset2_trans)
axins2.add_collection(p_inset3_trans)

#Annotate the inset plots
axins.annotate("Panel 1", (1410, 2300), color="w", size=20)
axins1.annotate("Panel 2", (2230, 2000), color="w", size=20)
axins2.annotate("Panel 3", (2010, 2410), color="w", size=20)

# Sets the ranges on the collections to their correct min and max values
if param == "T":
    plotting(param, mean_T, mean_T_trans)
if param == "n":
    plotting(param, mean_n, mean_n_trans)
if param == "N_SIO":
    plotting(param, mean_N_SIO, mean_N_SIO_trans)
if param == "N_SO":
    plotting(param, mean_N_SO, mean_N_SO_trans)
