# -*- coding: utf-8 -*-

import csv
import os
from multiprocessing import Pool
import psycopg2
import sys
import subprocess as sp
import time

# Disables numpy from being able to spawn parallel processes itself
#os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(1, "{0}/UCLCHEM/scripts/".format(os.getcwd()))
import plotfunctions

#from astropy import units as u
from decimal import Decimal
import numpy as np

from chainconsumer import ChainConsumer
import emcee as mc
import math
import random

import matplotlib.pyplot as plt

import config as config
import databasefunctions as db
import inference
import workerfunctions

def param_select(params):
    if params['vs']:
        params['vs'] = random.uniform(5, 30)
    if params['initial_dens']:
        params['initial_dens'] = random.uniform(3, 8)
    if params['b_field']:
        params['b_field'] = random.uniform(-6, -3) # B-field in gauss
    if params['crir']:
        params['crir'] = random.uniform(1, 2) # Cosmic ray ionisation rate (zeta in UCLCHEM)
    if params['isrf']:
        params['isrf'] = random.uniform(1, 2) # Interstellar radiation field  (radfield in UCLCHEM)

    return params


# Define constants
DIREC = os.getcwd()
RADEX_PATH = "{0}/../Radex".format(DIREC)
#RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"



if __name__ == '__main__':

    # Define the datafile
    datafile = "{0}/data/tabula-sio_intratio.csv".format(DIREC)

    # Define molecular data
    sio_data = {
        'T': [100., 1000.],
        'g_u': 15.0,
        'A_ul': 10**(-2.83461),
        'E_u': 58.34783,
        'Z': [72.3246, 962.5698]
    }

    so_data = {
        'T': [100., 1000.],
        'g_u': 17.0,
        'A_ul': 10**(-1.94660),
        'E_u': 62.14451,
        'Z': [197.515, 2*850.217] # From Splatalogue (https://www.cv.nrao.edu/php/splat/species_metadata_displayer.php?species_id=20)
    }

    # Declare the species that we're interested in
    relevant_species = ["SIO", "SO", "CH3OH", "OCS", "H2CS"]

    # Declare the database connections
    config_file = config.config_file(db_init_filename='database_archi.ini', section='shock_fit_results')
    bestfit_config_file = config.config_file(db_init_filename='database_archi.ini', section='shock_bestfit_conditions')

    # Set up the connection pools
    db_pool = db.dbpool(config_file)
    db_bestfit_pool = db.dbpool(bestfit_config_file)

    # Read in the whole data file containing sources and source flux
    with open(datafile, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()

    # Parse the data to a dict list
    observed_data = workerfunctions.parse_data(data, db_pool, db_bestfit_pool)
    
    # Filter the observed data to contain only those species that we can use
    # (normally limited by those with Radex data)
    filtered_data = workerfunctions.filter_data(observed_data, relevant_species)
    nWalkers = 500 # Number of random walkers to sample parameter space
    nDim = 5 # Number of dimensions within the parameters
    nSteps = int(1e4) # Number of steps per walker
    

    for obs in filtered_data:
        if (len(obs["species"]) >= 2 and "SIO" in obs["species"]) or \
                (len(obs["species"]) >= 2 and "SO" in obs["species"]) or \
                    (len(obs["species"]) >= 2 and "CH3OH" in obs["species"]) or \
                        (len(obs["species"]) >= 2 and "OCS" in obs["species"]) or \
                            (len(obs["species"]) >= 2 and "H2CS" in obs["species"]):

            if obs["source"] == "K13a" or obs["source"] == "G2":
                print("Source = {}".format(obs["source"]))
            else:
                continue

            # Checks to see whether the tables exists; if so, delete it
            if db.does_table_exist(db_pool=db_pool, table=obs["source"]):
                db.drop_table(db_pool=db_pool, table=obs["source"])
                #continue
            if db.does_table_exist(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"])):
                db.drop_table(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"]))
                #continue            
            
            # Store the column density string
            all_column_str = ""  #  Empty string to hold full string once complete
            for spec_indx, spec in enumerate(obs["species"]):
                if spec == "SIO" or spec == "SO" or spec == "CH3OH" or spec == "H2CS" or spec == "OCS":
                    column_str = "column_density_{0} REAL NOT NULL, ".format(
                        spec)

                    # Remove the comma on the final column density
                    if spec_indx == (len(obs["species"])-1):
                        column_str = column_str[:-2:]
                    all_column_str = all_column_str + column_str
                    
            # Define the commands necessary to create table in database (raw SQL string)
            commands = (
                """
                CREATE TABLE IF NOT EXISTS {0} (
                    id SERIAL PRIMARY KEY,
                    vs REAL NOT NULL,
                    dens REAL NOT NULL,
                    b_field REAL NOT NULL,
                    crir REAL NOT NULL,
                    isrf REAL NOT NULL
     	           );
                """.format(obs["source"]),
            )
            
            bestfit_commands = (
                """
                CREATE TABLE IF NOT EXISTS {0}_bestfit_conditions (
                    id SERIAL PRIMARY KEY,
                    species TEXT [] NOT NULL,
                    transitions TEXT [] NOT NULL,
                    vs REAL NOT NULL,
                    dens REAL NOT NULL,
                    b_field REAL NOT NULL,
                    crir REAL NOT NULL,
                    isrf REAL NOT NULL,
                    column_density DOUBLE PRECISION [] NOT NULL,
                    resolved_T REAL NOT NULL,
                    resolved_n REAL NOT NULL,
                    radex_flux DOUBLE PRECISION [] NOT NULL,
                    source_flux DOUBLE PRECISION [] NOT NULL,
                    source_flux_error DOUBLE PRECISION [] NOT NULL,
                    chi_squared DOUBLE PRECISION NOT NULL
                );
                """.format(obs["source"]),
            )

            # Create the tables
            db.create_table(db_pool=db_pool, commands=commands)
            db.create_table(db_pool=db_bestfit_pool, commands=bestfit_commands)
            
            # Determine the number of dimensions
            print("nDim={0}".format(nDim))

            # Define the column names for saving to the database
            # column_names = ["vs", "dens"] + ["column_density_{0}".format(spec) for spec in obs["species"]]
            column_names = ["vs", "dens", "b_field", "crir", "isrf"]

            print("Running the emcee sampler")
            sampler = mc.EnsembleSampler(nWalkers, nDim, inference.ln_likelihood_shock,
                            args=(obs, db_bestfit_pool, DIREC, RADEX_PATH), pool=Pool(24))
            pos = []

            # Select the parameters
            for i in range(nWalkers):
                params = param_select({
                    'vs': True,
                    'initial_dens': True,
                    'b_field': True,
                    'crir': True,
                    'isrf': True,
                })

                # Iterate through those lists to find entries that
                # are still marked as False (and not required)
                pos_list = [x for x in params.values() if x != False]

                # Append the correct params to the params list
                pos.append(pos_list)

            # Split the chain in to 100 chunks, each 1% of the total size and write out
            nBreak = int(nSteps/1e2)
            for counter in range(nBreak):
                sampler.reset()  # Reset the chain

                print("Running mcmc")

                # start from where we left off previously
                pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=False)

                for species in obs["species"]:
                    # Delete the Radex and UCLCHEM input and output files
                    inference.delete_radex_uclchem_io(species, DIREC)
                inference.delete_uclchem_io(DIREC)

                #chain is organized as chain[walker, step, parameter(s)]
                #chain = np.array(sampler.chain[:, :, :])
                chain = np.array(sampler.chain)

                for i in range(0, nWalkers):
                    for j in range(0, nBreak):
                        db.insert_shock_chain_data(
                            db_pool=db_pool, table=obs["source"], chain=chain[i][j], column_names=column_names)

                if counter == 0:
                    full_chain = np.vstack(chain)
                else:
                    full_chain = np.concatenate((full_chain, np.vstack(chain)))

                # Checks for convergence using ChainConsumer
                c = ChainConsumer()
                c.add_chain(full_chain, parameters=column_names,
                            walkers=nWalkers)

                # Convergence tests
                gelman_rubin = c.diagnostic.gelman_rubin(threshold=0.15)
                if gelman_rubin:
                    print("Chains have converged")
                    break
                else:
                    print("Chains have yet to converge")

                ''' 
                # Plot the UCLCHEM plots
                vs, initial_dens, b_field, crir, isrf = chain_results[0], chain_results[1], chain_results[2], chain_results[3], chain_results[4]
                uclchem_file = "{0}/UCLCHEM/output/data/v{1:.2}n1e{2:.2}z{3:.2}r{4:.2}b{5:.2}.dat".format(
                    DIREC, vs, initial_dens, crir, isrf, b_field)

                # read file
                times, dens, temp, abundances = plotfunctions.read_uclchem(uclchem_file, obs["species"])

                shock_model = {
                    "times": times,
                    "dens": dens,
                    "temp": temp,
                    "abundances": abundances
                }
                '''
                
            '''    
            print("Moving to plotting routine")
            print("Getting data from database")

            # Read the database to retrieve the data
            chains = db.get_chains(
                db_pool=db_pool, 
                table=obs["source"], 
                column_names=[
                    "vs",
                    "dens",
                    "b_field", 
                    "crir", 
                    "isrf"
                ]
            )

            # Determine the length of the first chain (assuming all chains are the same length)
            chain_length = len(chains)

            # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
            chain = np.array(chains[int(chain_length*0.2):])

            #Name params for chainconsumer (i.e. axis labels)
            param1 = "v$_{s}$ / km s$^{-1}$"
            param2 = "log(n$_{H}$) / cm$^{-3}$"
            param3 = "B$_{0}$ / G"
            param4 = "$\zeta$ / s$^{-1}$"
            param5 = "I$_{r}$"

            #Chain consumer plots posterior distributions and calculates
            #maximum likelihood statistics as well as Geweke test
            file_out = "{0}/radex-plots/new/corner_{1}.png".format(DIREC, obs["source"])
            file_out_walk = "{0}/radex-plots/new/walks/walk_{1}.png".format(DIREC, obs["source"])
            c = ChainConsumer() 
            try:
                c.add_chain(chain, parameters=[param1,param2,param3,param4,param5], walkers=nWalkers)
            except AssertionError:
                continue
            c.configure(color_params="posterior", cloud=True, usetex=True, summary=False) 
            fig = c.plotter.plot(filename=file_out, display=False)
            # fig.set_size_inches(6 + fig.get_size_inches())
            # summary = c.analysis.get_summary()

            fig_walks = c.plotter.plot_walks(filename=file_out_walk, display=False, plot_posterior=True)
        '''

