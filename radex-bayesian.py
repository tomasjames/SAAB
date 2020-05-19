# -*- coding: utf-8 -*-

import csv
import os
from multiprocessing import Pool
import psycopg2
import sys
import subprocess as sp

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
    if params['temp']:
        T = round(random.uniform(100, 2500), 2)
        params['temp'] = T
    if params['dens']:
        dens = round(random.uniform(3, 8), 2)
        params['dens'] = dens
    if params['N_sio']:
        N_sio = round(random.uniform(8, 16), 2)
        params['N_sio'] = N_sio
    if params['N_so']:
        N_so = round(random.uniform(8, 16), 2)
        params['N_so'] = N_so
    if params['N_ocs']:
        N_ocs = round(random.uniform(8, 16), 2)
        params['N_ocs'] = N_ocs

    return params


# Define constants
DIREC = os.getcwd()
# RADEX_PATH = "{0}/../Radex".format(DIREC)
RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"

print_results = True

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
    relevant_species = ["SIO", "SO", "OCS"]

    # Declare the database connections
    radex_config_file = config.config_file(db_init_filename='database.ini', section='radex_fit_results')
    bestfit_config_file = config.config_file(db_init_filename='database.ini', section='bestfit_conditions')

    # Set up the connection pools
    db_radex_pool = db.dbpool(radex_config_file)
    db_bestfit_pool = db.dbpool(bestfit_config_file)

    # Read in the whole data file containing sources and source flux
    with open(datafile, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()

    # Parse the data to a dict list
    observed_data = workerfunctions.parse_data(data, db_radex_pool, db_bestfit_pool)

    # Filter the observed data to contain only those species that we can use
    # (normally limited by those with Radex data)
    filtered_data = workerfunctions.filter_data(
        observed_data, relevant_species)
    
    # Determine the estimated column densities based on the temperature (and a number of 
    # other assumptions) and create the necessary databases
    physical_conditions = []
    for obs in observed_data[:8]:
        
        if (len(obs["species"]) >= 2 and "SIO" in obs["species"]) or \
                (len(obs["species"]) >= 2 and "SO" in obs["species"]) or \
                    (len(obs["species"]) >= 2 and "OCS" in obs["species"]):
            
            # Checks to see whether the tables exists; if so, delete it
            if db.does_table_exist(db_pool=db_radex_pool, table=obs["source"]):
                db.drop_table(db_pool=db_radex_pool, table=obs["source"])
            if db.does_table_exist(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"])):
                db.drop_table(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"]))
            
            # Store the column density string
            all_column_str = "" # Empty string to hold full string once complete
            for spec_indx, spec in enumerate(obs["species"]):
                if spec == "SIO" or spec == "SO" or spec == "OCS":
                    column_str = "column_density_{0} REAL NOT NULL, ".format(spec)
                    if spec_indx == (len(obs["species"])-1): # Remove the comma 
                        column_str = column_str[:-2:]
                    all_column_str = all_column_str + column_str
            
            # Define the commands necessary to create table in database (raw SQL string)
            commands = (
                """
                CREATE TABLE IF NOT EXISTS {0} (
                    id SERIAL PRIMARY KEY,
                    temp REAL NOT NULL,
                    dens REAL NOT NULL,
                    {1}
                );
                """.format(obs["source"], all_column_str),
            )
            
            bestfit_commands = (
                """
                CREATE TABLE IF NOT EXISTS {0}_bestfit_conditions (
                    id SERIAL PRIMARY KEY,
                    species TEXT [] NOT NULL,
                    transitions TEXT [] NOT NULL,
                    temp REAL NOT NULL,
                    dens REAL NOT NULL,
                    column_density DOUBLE PRECISION [] NOT NULL,
                    radex_flux DOUBLE PRECISION [] NOT NULL,
                    source_flux DOUBLE PRECISION [] NOT NULL,
                    source_flux_error DOUBLE PRECISION [] NOT NULL,
                    chi_squared DOUBLE PRECISION NOT NULL
                );
                """.format(obs["source"]),
            )

            # Create the tables
            db.create_table(db_pool=db_radex_pool, commands=commands)
            db.create_table(db_pool=db_bestfit_pool, commands=bestfit_commands)
            
            # Determine estimates of what the physical conditions should be
            physical_conditions.append(inference.param_constraints(obs, sio_data, so_data))
    
    nWalkers = 40 # Number of random walkers to sample parameter space
    nSteps = int(1e2) # Number of steps per walker
    
    #Set up MPI Pool
    pool = Pool(4)

    # Loop through the sources 
    for obs in observed_data[:8]:
        if (len(obs["species"]) >= 2 and "SIO" in obs["species"]) or \
             (len(obs["species"]) >= 2 and "SO" in obs["species"]) or \
                (len(obs["species"]) >= 2 and "OCS" in obs["species"]):

            print(obs["source"])
            print(obs["species"])
            # Empty lists for parameters and their keys
            pos, pos_keys = [], []

            # Empty flags to signify whether molecules are present
            # Assune False by default
            sio_flag, so_flag, ocs_flag = False, False, False

            # Determine which molecules are present and alter flags
            if "SIO" in obs["species"]:
                sio_flag = True
            if "SO" in obs["species"]:
                so_flag = True
            if "OCS" in obs["species"]:
                ocs_flag = True

            # Select the parameters
            for i in range(nWalkers):
                params = param_select({
                    'temp': True,
                    'dens': True,
                    'N_sio': sio_flag,
                    'N_so': so_flag,
                    'N_ocs': ocs_flag
                })

                # Convert the keys and values to lists
                pos_keys_list = list(params.keys())
                pos_list = list(params.values())

                # Iterate through those lists to find entries that
                # are still marked as False (and not required)
                for param_indx, param in enumerate(pos_list):
                    if param == False:
                        pos_list.pop(param_indx)
                        pos_keys_list.pop(param_indx)

                # Append the correct params to the params list
                pos.append(pos_list)
                pos_keys.append(pos_keys_list)

            # Determine the number of dimensions
            nDim = len(pos_keys[0])
            print("nDim={0}".format(nDim))

            # Run the sampler
            sampler = mc.EnsembleSampler(nWalkers, nDim, inference.ln_likelihood_radex, 
                args=(obs, bestfit_config_file, DIREC, RADEX_PATH), pool=pool)

            # Split the chain in to 100 chunks, each 1% of the total size and write out
            nBreak=int(nSteps/100)
            for counter in range(0, nSteps):
                sampler.reset() # Reset the chain
                print("Running mcmc")
                pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=False) #start from where we left off previously 

                for species in obs["species"]:
                    # Delete the Radex and UCLCHEM input and output files
                    inference.delete_radex_io(species, DIREC)
                
                #chain is organized as chain[walker, step, parameter(s)]
                chain = np.array(sampler.chain[:, :, :])
                for i in range(0, nWalkers):
                    for j in range(0, nBreak):
                        store = []
                        for k in range(0, nDim):
                            store.append(chain[i][j][k])
                        db.insert_radex_chain_data(db_pool=db_radex_pool, table=obs["source"], chain=store)

                sampler.reset()
            '''
            if print_results:
                print("Moving to plotting routine")
                print("Getting data from database")

                # Read the database to retrieve the data
                try:
                    chains = db.get_chains(
                        db_pool=db_radex_pool, 
                        table=obs["source"], 
                        column_names=[
                            "temp",
                            "dens",
                            "column_density_SIO",
                            "column_density_SO",
                            "column_density_OCS"
                        ]
                    )
                except:
                    chains = db.get_chains(
                        db_pool=db_radex_pool,
                        table=obs["source"],
                        column_names=[
                            "temp",
                            "dens",
                            "column_density_SIO",
                            "column_density_SO"
                        ]
                    )

                # Determine the length of the first chain (assuming all chains are the same length)
                try:
                    chain_length = len(chains)
                except TypeError:
                    continue

                # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
                chain = np.array(chains[int(chain_length*0.2):])

                #Name params for chainconsumer (i.e. axis labels)
                param1 = "T / K"
                param2 = "log(n$_{H}$) / cm$^{-3}$"
                param3 = "log(N$_{SiO}$) / cm$^{-2}$"
                param4 = "log(N$_{SO}$) / cm$^{-2}$"
                param5 = "log(N$_{OCS}$) / cm$^{-2}$"

                #Chain consumer plots posterior distributions and calculates
                #maximum likelihood statistics as well as Geweke test
                file_out = "{0}/radex-plots/new/corner_{1}.pdf".format(DIREC, obs["source"])
                file_out_walk = "{0}/radex-plots/new/walks/walk_{1}.pdf".format(DIREC, obs["source"])
                c = ChainConsumer() 
                c.add_chain(chain, parameters=[param1,param2,param3,param4], walkers=nWalkers)
                c.configure(color_params="posterior", cloud=True, usetex=True, summary=False) 
                fig = c.plotter.plot(filename=file_out, display=False)
                # fig.set_size_inches(6 + fig.get_size_inches())
                # summary = c.analysis.get_summary()

                #fig_walks = c.plotter.plot_walks(filename=file_out_walk, display=False, plot_posterior=True)
                fig.close()
            '''
    pool.close()
