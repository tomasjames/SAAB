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


def param_select():
    T = random.uniform(75, 1000)
    dens = random.uniform(3, 6)
    N_sio = random.uniform(11, 15)
    N_so = random.uniform(10, 14)

    return T, dens, N_sio, N_so


# Define constants
DIREC = os.getcwd()
# RADEX_PATH = "{0}/../Radex".format(DIREC)
RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"



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
    
    # Determine the estimated column densities based on the temperature (and a number of 
    # other assumptions) and create the necessary databases
    physical_conditions = []
    for obs in observed_data:
        '''
        if obs["species"] == ["SIO", "SO"]:

            # Checks to see whether the tables exists; if so, delete it
            if db.does_table_exist(db_pool=db_radex_pool, table=obs["source"]):
                db.drop_table(db_pool=db_radex_pool, table=obs["source"])
            if db.does_table_exist(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"])):
                db.drop_table(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"]))

            # Define the commands necessary to create table in database (raw SQL string)
            commands = (
                """
                CREATE TABLE IF NOT EXISTS {0} (
                    id SERIAL PRIMARY KEY,
                    temp REAL NOT NULL,
                    dens REAL NOT NULL,
                    column_density_SIO REAL NOT NULL,
                    column_density_SO REAL NOT NULL
                );
                """.format(obs["source"]),
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
        '''
    
    # continueFlag = False
    nWalkers = 40 # Number of random walkers to sample parameter space
    nDim = 4 # Number of dimensions within the parameters
    nSteps = int(1e3) # Number of steps per walker
    
    # Set up the backend for temporary chain storage
    # Don't forget to clear it in case the file already exists
    filename = "{0}/chains/chains.h5".format(DIREC)
    backend = mc.backends.HDFBackend(filename)
    backend.reset(nWalkers, nDim)

    #Set up MPI Pool
    pool = Pool(4)

    for obs in observed_data[:8]:
        if obs["species"] != ["SIO", "SO"]:
            continue
        else:
            '''
            sampler = mc.EnsembleSampler(nWalkers, nDim, inference.ln_likelihood_radex, 
                args=(obs, db_bestfit_pool, DIREC, RADEX_PATH), backend=backend, pool=pool)
            pos = []
            
            # Select the parameters
            for i in range(nWalkers):
                T, dens, N_sio, N_so = param_select()
                pos.append([T, dens, N_sio, N_so])

            print(obs["source"])

            # Split the chain in to 100 chunks, each 1% of the total size and write out
            nBreak=int(nSteps/10)
            for counter in range(0, nBreak):
                sampler.reset() # Reset the chain
                print("Running mcmc")
                pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 

                for data in observed_data:
                    # Delete the Radex and UCLCHEM input and output files
                    inference.delete_radex_io(data["species"], DIREC)
                    # inference.delete_uclchem_io(DIREC)
                
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
            print("Moving to plotting routine")
            print("Getting data from database")

            # Read the database to retrieve the data
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
            chain_length = len(chains)

            # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
            chain = np.array(chains[int(chain_length*0.2):])

            # Round the data
            for indy, entry_y in enumerate(chain):
                for indx, entry_x in enumerate(entry_y):
                    chain[indy][indx] = round(entry_x, 1)

            #Name params for chainconsumer (i.e. axis labels)
            param1 = "T / K"
            param2 = "log(n$_{H}$) / cm$^{-3}$"
            param3 = "log(N$_{SiO}$) / cm$^{-2}$"
            param4 = "log(N$_{SO}$) / cm$^{-2}$"

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
    pool.close()
    
