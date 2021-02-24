from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy import wcs

from chainconsumer import ChainConsumer

import csv
import numpy as np
from scipy import stats
import matplotlib
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
pointing = "K"

m_H_kg = 1.6735326381e-27
m_H2_kg = 2*m_H_kg
m_H2_g = m_H2_kg*1e3
M_solar_kg = 1.989e+30
M_solar_g = (1.989e+30)*1e3
sgrA_dist_pc = 7861.2597
sgrA_dist_cm = sgrA_dist_pc*3.086e+18

# Define lists
T, T_trans, n, n_trans = [], [], [], []
T_upper, T_lower, n_upper, n_lower = [], [], [], []
N_SIO, N_SIO_trans, N_SO, N_SO_trans = [], [], [], []

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

fig1 = plt.figure(figsize=(18, 18))

source_name = ""

for indx, entry in enumerate(data[1:]):  # data[0] is the header row

    if entry[0][0] != pointing or entry[0] == source_name:
        continue

    source_name = entry[0]
    print("source_name={0}".format(source_name))

    # Get source radius
    radius_angle = (float(entry[3])*np.deg2rad(0.37/3600))/2
    radius_pc = radius_angle*7861.2597
    radius_cm = radius_pc*3.086e+18  # Distance to Sgr A* from pc to cm

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

    # Ensures only filled data continues
    if best_fit_data == None:
        continue

    # G2 can have a larger chain - fixes this
    if entry[0] == "G2":
        best_fit_data = best_fit_data[:int(235000)]

    # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
    chain = np.array(best_fit_data[int(len(best_fit_data)*0.2):])
    
    c.add_chain(chain, parameters=[
        "temp",
        "dens",
        "column_density_sio",
        "column_density_so"
    ],
        walkers=500, name=entry[0])

    # Get summary
    summary = c.analysis.get_summary()

    # Store the mean temperature for sources of interest
    if summary["temp"][0] == None or summary["dens"][0] == None:
        T_trans.append(summary["temp"][1])
        n_trans.append(summary["dens"][1])
        N_SIO_trans.append(summary["column_density_sio"][1])
        N_SO_trans.append(summary["column_density_so"][1])
    else:
        T.append(summary["temp"][1])
        T_lower.append(summary["temp"][0])
        T_upper.append(summary["temp"][2])

        n.append(summary["dens"][1])
        n_lower.append(summary["dens"][0])
        n_upper.append(summary["dens"][2])

        N_SIO.append(summary["column_density_sio"][1])
        N_SO.append(summary["column_density_so"][1])

    # Determine the object's mass
    rho = 10**(summary["dens"][1])*m_H2_g # g/cm-3
    M_uniform_g = ((4/3)*np.pi*(radius_cm**3)*rho) # in g
    M_uniform_M0 = M_uniform_g/M_solar_g
    # print("M_uniform_M0 = {0}".format(M_uniform_M0))

    # Determine the objects virial mass
    M_vir_M0 = (200*radius_pc*float(entry[-2])**2) # in solar mass
    M_vir_g = M_vir_M0*M_solar_g
    # print("M_vir_M0 = {0}".format(M_vir_M0))

    # Determine the object's thermal pressure
    p_therm_kb = 10**(summary["dens"][1]) * (summary["temp"][1])
    # print("p_therm_kb = {0}".format(p_therm_kb))

    # Determine the object's turbulent pressure
    p_turb_kb = (rho*(float(entry[-2])*1e5)**2)/(1.38e-16)
    # print("p_turb_kb = {0}".format(p_turb_kb))

    if p_therm_kb > p_turb_kb/3:
        print("Thermal pressure dominates")
    else:
        print("Turbulent pressure dominates")
    
    if M_uniform_M0 > M_vir_M0/3:
        print("Star forming gas! :D")
    else:
        print("Not star forming gas :(")

norm = matplotlib.colors.Normalize(
    vmin=min(N_SIO+N_SIO_trans), vmax=max(N_SIO+N_SIO_trans), clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
color = np.array([(mapper.to_rgba(v)) for v in n])
color_trans = np.array([(mapper.to_rgba(v)) for v in n_trans])

# Plot the whole data to establish the colorbar
sc = plt.scatter(T+T_trans, n+n_trans, c=N_SIO+N_SIO_trans)
clb = plt.colorbar(sc, orientation=("vertical"))
clb.set_label(label="N"+r"$_{\mathrm{SiO}}$"+"[cm$^{-2}$]", fontsize=20)
clb.ax.tick_params(labelsize=18)

#loop over each data point to plot
for x, y, col in zip(T_trans, n_trans, color_trans):
    plt.plot(x, y, 'o', color=col)

for x, y, x_lower, x_upper, y_lower, y_upper, col in zip(T, n, T_lower, T_upper, n_lower, n_upper, color):
    # plt.plot(x, y, 'o', color=col)
    plt.plot(np.linspace(x, x, 2),np.linspace(y_lower, y_upper, 2), lw=1, color=col)
    plt.plot(np.linspace(x_lower, x_upper, 2), np.linspace(y, y, 2), lw=1, color=col)

plt.xlabel("T [K]", fontsize=20)
plt.ylabel("n [cm$^{-3}$]", fontsize=20)
plt.savefig("{0}/data/images/n_vs_T.png".format(DIREC))
plt.close()
