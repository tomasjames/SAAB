from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy.wcs import WCS

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import os

# plt.style.use(astropy_mpl_style)



# Define constants
DIREC = os.getcwd()

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
wcs = WCS(observation.header)

############################## Plot the data ############################
# Initialize figure
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85], projection=wcs, slices=('x', 'y', 0))

# # Plot the data
plot = ax.imshow(image_data, interpolation="nearest", origin="lower")
fig.colorbar(plot)
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")

# Add the grid
g = ax.coords.grid(alpha=0.25, color='white')

# Zoom in by a factor of 2.5 and place on the upper left
axins = zoomed_inset_axes(ax, 2.5, loc=2)
axins.imshow(image_data, interpolation="nearest", origin="lower")
plt.yticks(visible=False)
plt.xticks(visible=False)

ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])

# Add on the sources
for indx, entry in enumerate(data[1:]):  # data[0] is the header row
    
    # Only need to pull out the coordinates of each source 
    coords = SkyCoord(
        ra=Angle('17h45m{0}s'.format(entry[1])), dec=Angle('-29d00m{0}s'.format(entry[2])), frame='fk4')
    print(coords.ra.hms, coords.dec.deg)

    ax1.scatter(coords.ra.hour, coords.dec.deg, marker='+',
                label="{0}".format(entry[0]))
    axins.scatter(coords.ra.hour, coords.dec.deg, marker='+',
                label="{0}".format(entry[0]))

ax1.patch.set_alpha(.0)
ax1.axis('off')

x1, x2, y1, y2 = 2100, 2450, 3100, 4100  # specify the limits
axins.set_xlim(x1, x2)  # apply the x-limits
axins.set_ylim(y1, y2)  # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
fig.savefig('data/images/HCN.pdf')
