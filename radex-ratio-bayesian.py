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
from inference import ln_likelihood, reset_data_dict, linewidth_conversion, delete_radex_io
from correlation import autocorr_new


def param_select():
    vs = random.uniform(30, 60)
    initial_dens = random.uniform(3, 6)

    return vs, initial_dens


# Define constants
DIREC = os.getcwd()
# RADEX_PATH = "{0}/../Radex".format(DIREC)
RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"



if __name__ == '__main__':

    # Define the datafile
    datafile = "{0}/data/tabula-sio_intratio.csv".format(DIREC)

    # Define dictionary to store data
    data_storage = reset_data_dict()

    # And a list to store the data dictionaries
    observed_data = []

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

    # Declare the database connection 
    config_file = config.config_file(db_init_filename='database.ini', section='postgresql')
    
    # Read in the whole data file containing sources and source flux
    with open(datafile, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()

    # Declare a variable to help track the concurrent sources
    source_indx = 0

    # Extract the columns from the data and insert in to data list
    for indx, entry in enumerate(data[1:]): # data[0] is the header row
        source_name = entry[0] # The source name
        sample_size = entry[3] # The sample size (in beams)
        species = entry[4][0:entry[4].find(" ")].upper() # The molecule of interest
        transitions = entry[4][entry[4].find(" ")+1:].replace("–", "--").replace("(", "_").replace(")", "") # The transition of the molecule
        source_flux_dens_Jy = (float(entry[5][0:entry[5].find("±")])/1e3) # The observed flux density (Jy)
        source_flux_dens_error_Jy = (float(entry[5][entry[5].find("±")+1:])/1e3) # The observed flux error (Jy)
        linewidths = (float(entry[7])*1e3) # The linewidth (m/s)
        
        # Check to see whether this is the first data entry 
        # Or whether the source already exists within the conditioned dictionary
        if indx == 0 or observed_data[len(observed_data)-1]['source'] != source_name:
            source_indx += 1
            data_storage['source'] = source_name
            data_storage['sample_size'] = sample_size
            data_storage['species'].append(species)
            data_storage['transitions'].append(transitions)
            data_storage['source_flux_dens_Jy'].append(float(source_flux_dens_Jy))
            data_storage['source_flux_dens_error_Jy'].append(float(source_flux_dens_error_Jy))
            data_storage['linewidths'].append(float(linewidths))
            
            if species == "SIO":
                transition_freq = 303.92696000*1e9
            elif species == "SO":
                transition_freq = 304.07791400*1e9
            data_storage['transition_freqs'].append(transition_freq)

            linewidth_f = linewidth_conversion(linewidths, transition_freq)
            
            data_storage["source_flux"].append(
                (data_storage["source_flux_dens_Jy"][0]*1e-23)*linewidth_f
            )
            data_storage["source_flux_error"].append(
                (data_storage["source_flux_dens_error_Jy"][0]*1e-23)*linewidth_f
            )
            
            # Checks to see whether the table exists; if so, delete it
            if db.does_table_exist(db_params=config_file, table=source_name):
                db.drop_table(db_params=config_file, table=source_name)
            
            observed_data.append(data_storage)
            data_storage = reset_data_dict() # Reset the data dict
        else:
            
            # Define the commands necessary to create table in database (raw SQL string)
            commands = (
                """
                CREATE TABLE IF NOT EXISTS {0} (
                    id SERIAL PRIMARY KEY,
                    vs REAL NOT NULL,
                    initial_dens REAL NOT NULL
                )
                """.format(source_name),
            )
            # Create the table
            db.create_table(db_params=config_file, commands=commands)
            
            # Append the data to the pre-existing entry in the dict-list
            observed_data[source_indx-1]['species'].append(species)
            observed_data[source_indx-1]['transitions'].append(transitions)
            observed_data[source_indx-1]['source_flux_dens_Jy'].append(source_flux_dens_Jy)
            observed_data[source_indx-1]['source_flux_dens_error_Jy'].append(source_flux_dens_error_Jy)
            observed_data[source_indx-1]['linewidths'].append(linewidths)

            if species == "SIO":
                transition_freq = 303.92696000*1e9
            elif species == "SO":
                transition_freq = 304.07791400*1e9
            observed_data[source_indx-1]['transition_freqs'].append(transition_freq)

            linewidth_f = linewidth_conversion(linewidths, transition_freq)
            
            observed_data[source_indx-1]["source_flux"].append(
                (observed_data[source_indx-1]["source_flux_dens_Jy"][0]*1e-23)*linewidth_f
            )
            observed_data[source_indx-1]["source_flux_error"].append(
                (observed_data[source_indx-1]["source_flux_dens_error_Jy"][0]*1e-23)*linewidth_f
            )
    
    '''
    # Determine the estimated column densities based on the temperature (and a number of 
    # other assumptions)
    physical_conditions = param_constraints(observed_data, sio_data, so_data)
    '''

    continueFlag = False
    nWalkers = 4 # Number of random walkers to sample parameter space
    nDim = 2 # Number of dimensions within the parameters
    nSteps = int(1e1) # Number of steps per walker
    
    # Set up the backend for temporary chain storage
    # Don't forget to clear it in case the file already exists
    filename = "{0}/chains/chains.h5".format(DIREC)
    backend = mc.backends.HDFBackend(filename)
    backend.reset(nWalkers, nDim)

    autocorr_list = [[], [], [], []]

    for obs in observed_data[:8]:
        if obs["species"] != ["SIO", "SO"]:
            continue
        else:
            sampler = mc.EnsembleSampler(nWalkers, nDim, ln_likelihood, 
                args=(obs, DIREC, RADEX_PATH,), backend=backend, pool=Pool())
            pos = []
            
            if not continueFlag:
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
                    # Delete the Radex input and output files
                    delete_radex_io(data["species"], DIREC)
                
                #chain is organized as chain[walker, step, parameter(s)]
                chain = np.array(sampler.chain[:, :, :])
                for i in range(0, nWalkers):
                    for j in range(0, nBreak):
                        store = []
                        for k in range(0, nDim):
                            store.append(chain[i][j][k])
                        db.insert_chain_data(db_params=config_file, table=obs["source"], chain=store)

                sampler.reset()
            
            print("Moving to plotting routine")
            print("Getting data from database")

            # Read the database to retrieve the data
            print(config_file)
            print(obs["source"])
            print([
                "vs",
                "initial_dens",
            ])
            chains = db.get_chains(
                db_params=config_file, 
                table=obs["source"], 
                column_names=[
                    "vs",
                    "initial_dens"
                ]
            )

            # Determine the length of the first chain (assuming all chains are the same length)
            chain_length = len(chains)

            # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
            # TODO Add Geweke test to ensure convergence has occured outside of burn-in period
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
