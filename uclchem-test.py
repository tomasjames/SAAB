import os
import sys

sys.path.insert(1, "{0}/UCLCHEM/".format(os.getcwd()))

import workerfunctions

DIREC = os.getcwd()

velocities = np.linspace(30, 60, 4)
densities = [1e3, 1e4, 1e5, 1e6]

for vs in velocities:
    for initial_dens in densities:
        # Determine the dissipation length and identify
        # the point that it begins (max_temp_indx)
        dlength = 12.0*3.08e18*vs/initial_dens

        # Convert to time
        t_length = (dlength/(vs*1e5))/(60*60*24*365)
        times, dens, temp, abundances, coldens = workerfunctions.run_uclchem(vs, initial_dens, t_length, DIREC)
