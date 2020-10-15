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
    vs = random.uniform(10, 30)
    initial_dens = random.uniform(3, 6)

    return vs, initial_dens


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
        'Z': [72.3246, 2*480.8102]
    }

    so_data = {
        'T': [100., 1000.],
        'g_u': 17.0,
        'A_ul': 10**(-1.94660),
        'E_u': 62.14451,
        'Z': [197.515, 2*850.217] # From Splatalogue (https://www.cv.nrao.edu/php/splat/species_metadata_displayer.php?species_id=20)
    }

    # Declare the database connections
    config_file = config.config_file(db_init_filename='database.ini', section='postgresql')
    bestfit_config_file = config.config_file(db_init_filename='database.ini', section='bestfit_conditions')

    # Set up the connection pools
    db_pool = db.dbpool(config_file)
    db_bestfit_pool = db.dbpool(bestfit_config_file)

    # Read in the whole data file containing sources and source flux
    with open(datafile, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()

    observed_data = workerfunctions.parse_data(data, db_pool, db_bestfit_pool)
    
    '''
    # Determine the estimated column densities based on the temperature (and a number of 
    # other assumptions)
    physical_conditions = param_constraints(observed_data, sio_data, so_data)
    '''

    # continueFlag = False
    nWalkers = 8 # Number of random walkers to sample parameter space
    nDim = 2 # Number of dimensions within the parameters
    nSteps = int(1e2) # Number of steps per walker
    
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

            sampler = mc.EnsembleSampler(nWalkers, nDim, inference.ln_likelihood, 
                args=(obs, db_bestfit_pool, DIREC, RADEX_PATH), backend=backend, pool=pool)
            pos = []
            
            # Select the parameters
            for i in range(nWalkers):
                vs, initial_dens = param_select()
                pos.append([vs, initial_dens])

            print(obs["source"])

            # Split the chain in to 100 chunks, each 1% of the total size and write out
            nBreak=int(nSteps/10)
            for counter in range(0, nBreak):
                sampler.reset() # Reset the chain
                print("Running mcmc")
                pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=False) #start from where we left off previously 

                for data in observed_data:
                    # Delete the Radex and UCLCHEM input and output files
                    inference.delete_radex_io(data["species"], DIREC)
                    inference.delete_uclchem_io(DIREC)
                
                #chain is organized as chain[walker, step, parameter(s)]
                chain = np.array(sampler.chain[:, :, :])
                for i in range(0, nWalkers):
                    for j in range(0, nBreak):
                        store = []
                        for k in range(0, nDim):
                            store.append(chain[i][j][k])
                        db.insert_chain_data(db_pool=db_pool, table=obs["source"], chain=store)

                sampler.reset()
            
            print("Moving to plotting routine")
            print("Getting data from database")

            # Read the database to retrieve the data
            chains = db.get_chains(
                db_pool=db_pool, 
                table=obs["source"], 
                column_names=[
                    "vs",
                    "initial_dens"
                ]
            )

            # Determine the length of the first chain (assuming all chains are the same length)
            chain_length = len(chains)

            # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
            chain = np.array(chains[int(chain_length*0.2):])

            #Name params for chainconsumer (i.e. axis labels)
            param1 = "v$_{s}$ / km s$^{-1}$"
            param2 = "log(n$_{H}$) / cm$^{-3}$"

            #Chain consumer plots posterior distributions and calculates
            #maximum likelihood statistics as well as Geweke test
            file_out = "{0}/radex-plots/new/corner_{1}.png".format(DIREC, obs["source"])
            file_out_walk = "{0}/radex-plots/new/walks/walk_{1}.png".format(DIREC, obs["source"])
            c = ChainConsumer() 
            c.add_chain(chain, parameters=[param1,param2], walkers=nWalkers)
            c.configure(color_params="posterior", cloud=True, usetex=True, summary=False) 
            fig = c.plotter.plot(filename=file_out, display=False)
            # fig.set_size_inches(6 + fig.get_size_inches())
            # summary = c.analysis.get_summary()

            fig_walks = c.plotter.plot_walks(filename=file_out_walk, display=False, plot_posterior=True)
    pool.close()
