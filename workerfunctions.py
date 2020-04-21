import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# This is so that the code can execute from a directory
# above the UCLCHEM directory
sys.path.insert(1, "{0}/UCLCHEM/".format(os.getcwd()))
sys.path.insert(1, "{0}/UCLCHEM/scripts/".format(os.getcwd()))

import numpy as np

import uclchem
import plotfunctions



def run_uclchem(vs, n, t_evol, DIREC):

    # Define the table name
    file_name = "phase1-n{0:.2E}".format(n)

    # Set r_out in order to set the extinction correctly
    # within UCLCHEM
    if n <= 1e2:
        r_out = 5.3
    elif 1e2 < n <= 1e3:
        r_out = 2.6
    elif 1e3 < n <= 1e4:
        r_out = 0.5
    elif 1e4 < n <= 1e5:
        r_out = 0.25
    else:
        r_out = 0.05

    # Check if phase 1 exists (as n is the only important factor here)
    try:
        # db.does_table_exist(db_params=config_file, table=file_name)
        open("{0}/UCLCHEM/output/start/{1}.dat".format(DIREC, file_name))
    except FileNotFoundError:
        uclchem.general(
            {
                "initialDens": 1e2,
                "finalDens": n,
                "rout": r_out,
                "phase": 1,
                "readAbunds": 0,
                "abundFile": "{0}/UCLCHEM/output/start/{1}.dat".format(DIREC, file_name),
                "outputFile": "{0}/UCLCHEM/output/start/full_output_{1}.dat".format(DIREC, file_name)
            },
        ["SIO", "SO"])
    finally:
        #Run UCLCHEM
        uclchem.general(
            {
                "initialDens": 2*n,
                "finalDens": n,
                "finalTime": t_evol,
                "vs": vs,
                "rout": r_out,
                "phase": 2,
                "readAbunds": 1,
                "abundFile": "{0}/UCLCHEM/output/start/{1}.dat".format(DIREC, file_name),
                "outputFile": "{0}/UCLCHEM/output/data/v{1:.2E}n{2}.dat".format(DIREC, vs, n)
            },
            ["SIO", "SO"])
    
        print("UCLCHEM run complete")

        times, dens, temp, abundances = plotfunctions.read_uclchem(
            "{0}/UCLCHEM/output/data/v{1:.2E}n{2}.dat".format(DIREC, vs, n), ["SIO", "SO"])
    
        # Determine the H column density through the shock
        coldens = [r_out*(3e18)*n_i for n_i in dens]

        return {"times": times,
                "dens": dens,
                "temp": temp,
                "abundances": abundances,
                "H_coldens": coldens}


'''
def read_shock_data(filename):
    a = open(filename).read()
    a = a.split('\n')
    
    time, dens, temp, vel, av, N = ([] for i in range(6))
    
    with open(filename) as file:
        for line in file:
            bits = line.split()
            #find time line
            if 'age' in bits:
                time.append(float(bits[-2].replace('D', 'E')))
            #read another line for dens
            if 'density' in bits:
                densi = float(bits[-2].replace('D', 'E'))
                if densi == 0.0:
                    densi = 1e-10
                dens.append(densi)
            if "temp" in bits:
                tempi = float(bits[-2].replace('D', 'E'))
                temp.append(tempi)
            if "velocity" in bits:
                vel.append(float(bits[-2].replace('D', 'E')))
            if "extinction" in bits:
                av.append(float(bits[-2].replace('D', 'E')))
            if "column" in bits:
                N.append(float(bits[-2].replace('D', 'E')))

    return time, dens, temp, vel, av, N
'''


def resolved_quantity(density, measure, t):
    
    for indx in range(1, len(density)):
        dt = t[indx] - t[indx-1] 
        numerator =+ density[indx]*measure[indx]*dt
        denominator =+ density[indx]*dt

    quantity = numerator/denominator
    return quantity


'''
def time_distance_transform(vel, times):
    # Transform time in to distance (with 0 based array to account for the fact that
    # distance begins from 0)
    distance = [0]

    for i in range(1, len(times)):
        distance.append((vel[i]*1e5)*(times[i] - times[i-1]) * (60*60*24*365) + distance[i-1])

    return distance
'''
