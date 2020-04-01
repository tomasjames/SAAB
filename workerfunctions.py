import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# This is so that the code can execute from a directory
# below the UCLCHEM directory
sys.path.insert(1, "{0}/UCLCHEM/".format(os.getcwd()))
sys.path.insert(1, "{0}/UCLCHEM/scripts/".format(os.getcwd()))

import numpy as np

import uclchem
import plotfunctions




def run_uclchem(vs, n, DIREC):

    # Define the table name
    file_name = "phase1-n{0}".format(n)

    # Check if phase 1 exists (as n is the only important factor here)
    try:
        # db.does_table_exist(db_params=config_file, table=file_name)
        open("{0}/UCLCHEM/output/start/{1}.txt".format(DIREC, file_name))
    except FileNotFoundError:
        uclchem.general(
            {
                "initialDens": 1e2,
                "finalDens": n,
                "phase": 1,
                "readAbunds": 0,
                "abundFile": "{0}/UCLCHEM/output/start/{1}.txt".format(DIREC, file_name),
                "outputFile": "{0}/UCLCHEM/output/start/full_output_{1}.txt".format(DIREC, file_name)
            },
        ["SIO", "SO"])
    finally:
        #Run UCLCHEM
        uclchem.general(
            {
                "initialDens": 2*n,
                "finalDens": n,
                "vs": vs,
                "phase": 2,
                "readAbunds": 1,
                "abundFile": "{0}/UCLCHEM/output/start/{1}.txt".format(DIREC, file_name),
                "outputFile": "{0}/UCLCHEM/output/data/v{1:.2E}n{2}.txt".format(DIREC, vs, n)
            },
            ["SIO", "SO"])
    
        time, dens, temp, abundances = plotfunctions.read_uclchem(
            "{0}/UCLCHEM/output/data/v{1:.2E}n{2}.txt".format(DIREC, vs, n), ["SIO", "SO"])
    
    return time, dens, temp, abundances


def resolved_quantity(density, measure, x):
    
    for indx in range(1, len(density)):
        dx = x[indx] - x[indx-1] 
        numerator =+ density[indx]*measure[indx]*dx
        denominator =+ density[indx]*dx

        quantity = numerator/denominator
    return quantity
