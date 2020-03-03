#!/usr/bin/python

import csv
import glob, os
from multiprocessing import Pool
import psycopg2
import sys
import subprocess as sp

from astropy import units as u
from decimal import Decimal
import numpy as np

from chainconsumer import ChainConsumer
import emcee as mc
import random

import matplotlib.pyplot as plt

import config as config
import databasefunctions as db
from inference import *


def param_select():
    T = random.uniform(10, 500)
    dens = random.uniform(2, 6)
    N_sio = random.uniform(11, 14)
    N_so = random.uniform(10, 13)
    
    return T, dens, N_sio, N_so


# Define constants
DIREC = os.getcwd()
# RADEX_PATH = "{0}/../Radex".format(DIREC)
RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"



if __name__ == '__main__':

    # Define Farhad's data (temporary - need to compute from saved file)
    observed_data = {
        'species': ["SIO", "SO"], # Species of interest
        'transitions': ["7--6", "8_7--7_6"], # Transitions of interest
        'transition_freqs': [303.92696000, 304.07791400], # Transition frequencies according to Splatalogue in GHz
        'linewidths': [11.3, 9.23], # Linewidths for the transitions in km/s
        'source_flux_dens_Jy': [14.3/1e3, 18.5/1e3], # Flux of species in Jy (original in mJy)
        'source_flux_dens_error_Jy': [2.9/1e3, 3.6/1e3], # Error for each flux in Jy (original in mJy)
        'source_flux': [],
        'source_flux_error': []
    }
    
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
    
    # Connecting to the database using 'connect()' method
    config_file = config.config_file(db_init_filename='database.ini', section='postgresql')

    # Define the commands necessary to create table in database (raw SQL string)
    commands = (
        """
        CREATE TABLE IF NOT EXISTS chain_storage (
            id SERIAL PRIMARY KEY,
            T REAL NOT NULL,
            dens REAL NOT NULL,
            N_SIO REAL NOT NULL,
            N_SO REAL NOT NULL
        )
        """,
    )

    # Checks to see whether the table exists; if so, delete it
    table = "chain_storage"
    if db.does_table_exist(db_params=config_file, table=table):
        db.drop_table(db_params=config_file, table=table)
    
    # Create the table
    db.create_table(db_params=config_file, commands=commands)
    
    # Convert flux density to flux
    for indx in range(len(observed_data["linewidths"])):
        dv = observed_data["linewidths"][indx]*1e3
        transition_freq = observed_data["transition_freqs"][indx]*1e9

        linewidth_f = linewidth_conversion(dv, transition_freq)
        
        observed_data["source_flux"].append(
            (observed_data["source_flux_dens_Jy"][indx]*1e-23)*linewidth_f
        )
        observed_data["source_flux_error"].append(
            (observed_data["source_flux_dens_error_Jy"][indx]*1e-23)*linewidth_f
        )
    '''
    # Determine the estimated column densities based on the temperature (and a number of 
    # other assumptions)
    physical_conditions = param_constraints(observed_data, sio_data, so_data)
    '''
    continueFlag = False
    nWalkers = 12 # Number of random walkers to sample parameter space
    nDim = 4 # Number of dimensions within the parameters
    nSteps = int(1e4/nWalkers) # Number of steps per walker
    
    sampler = mc.EnsembleSampler(nWalkers, nDim, ln_likelihood, args=[observed_data], pool=Pool())
    
    pos = []
    
    if not continueFlag:
        for i in range(nWalkers):
            T, dens, N_sio, N_so = param_select()
            pos.append([T, dens, N_sio, N_so])

    # Split the chain in to 10 chunks, each 10% of the total size and write out
    nBreak=int(nSteps/10)
    for counter in range(0, 10):
        sampler.reset() # Reset the chain
        try: 
            pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 
        except ValueError:
            pos.pop() # Removes errorneous combination using pop()
            # Reselect values
            T, dens, N_sio, N_so = param_select()
            pos.append([T, dens, N_sio, N_so])
            pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 
        
        # Delete the Radex input and output files
        delete_radex_io(observed_data["species"])
        
        chain = np.array(sampler.chain[:,:,:]) #get chain for writing
        # chain = chain.astype(float)
        #chain is organized as chain[walker,step,parameter]
        for i in range(0, nWalkers):
            for j in range(0, nBreak):
                store = []
                for k in range(0, nDim):
                    store.append(chain[i][j][k])
                db.insert_chain_data(db_params=config_file, table="chain_storage", chain=store)

        sampler.reset()
    
    # Read the database to retrieve the data
    chains = db.get_chains(
        db_params=config_file, 
        table=table, 
        column_names=[
            "T",
            "dens",
            "N_SIO",
            "N_SO"
        ]
    )

    # Determine the length of the first chain (assuming all chains are the same length)
    chain_length = len(chains)

    # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
    # And concatenate the chain so as chain is contiguous array
    # TODO Add Geweke test to ensure convergence has occured outside of burn-in period
    chain = np.array(chains[int(chain_length*0.2):])
    # chain = np.concatenate(chains)

    #Name params for chainconsumer (i.e. axis labels)
    param1 = "T / K"
    param2 = "log(n$_{H}$) / cm$^{-3}$"
    param3 = "log(N$_{SiO}$) / cm$^{-2}$"
    param4 = "log(N$_{SO}$) / cm$^{-2}$"

    #Chain consumer plots posterior distributions and calculates
    #maximum likelihood statistics as well as Geweke test
    file_out = "{0}/radex-plots/corner.jpeg".format(DIREC)
    c = ChainConsumer() 
    c.add_chain(chain, parameters=[param1,param2,param3,param4], walkers=nWalkers)
    c.configure(color_params="posterior", cloud=True, usetex=True, summary=False) 
    fig = c.plotter.plot(filename=file_out, display=False)
    # fig.set_size_inches(6 + fig.get_size_inches())
    # summary = c.analysis.get_summary()
    