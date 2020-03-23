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
    T = random.uniform(75, 500)
    dens = random.uniform(3, 6)
    N_sio = random.uniform(11, 14)
    N_so = random.uniform(10, 13)
    
    return T, dens, N_sio, N_so


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

            '''
            # Checks to see whether the table exists; if so, delete it
            if db.does_table_exist(db_params=config_file, table=source_name):
                db.drop_table(db_params=config_file, table=source_name)
            '''
            observed_data.append(data_storage)
            data_storage = reset_data_dict() # Reset the data dict
        else:
            '''
            # Define the commands necessary to create table in database (raw SQL string)
            commands = (
                """
                CREATE TABLE IF NOT EXISTS {0} (
                    id SERIAL PRIMARY KEY,
                    T REAL NOT NULL,
                    dens REAL NOT NULL,
                    N_SIO REAL NOT NULL,
                    N_SO REAL NOT NULL
                )
                """.format(source_name),
            )
            # Create the table
            db.create_table(db_params=config_file, commands=commands)
            '''
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
    nWalkers = 120 # Number of random walkers to sample parameter space
    nDim = 4 # Number of dimensions within the parameters
    nSteps = int(1e2) # Number of steps per walker
    
    # Set up the backend for temporary chain storage
    # Don't forget to clear it in case the file already exists
    filename = "{0}/chains/chains.h5".format(DIREC)
    backend = mc.backends.HDFBackend(filename)
    backend.reset(nWalkers, nDim)

    autocorr_list = [[], [], [], []]

    for obs in observed_data:
        with Pool() as pool:
            if obs["species"] != ["SIO", "SO"]:
                continue
            else:    
                '''
                sampler = mc.EnsembleSampler(nWalkers, nDim, ln_likelihood, 
                    args=(obs, DIREC, RADEX_PATH,), backend=backend, pool=pool)
                pos = []
                
                if not continueFlag:
                    for i in range(nWalkers):
                        T, dens, N_sio, N_so = param_select()
                        pos.append([T, dens, N_sio, N_so])

                print(obs["source"])

                # Split the chain in to 10 chunks, each 10% of the total size and write out
                nBreak=int(nSteps/10)
                for counter in range(0, nBreak):
                    sampler.reset() # Reset the chain
                    pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 

                    for data in observed_data:
                        # Delete the Radex input and output files
                        delete_radex_io(data["species"], DIREC)
                    
                    # Check for convergence on the dimensions
                    # We'll track how the average autocorrelation time estimate changes
                    autocorr = np.empty(nBreak)
                    # Set the old autocorrelation time to some arbitrarily high value
                    old_tau = np.inf
                    
                    for dimension in range(nDim):
                        chain = np.array(sampler.chain[:,:,dimension].T) #get dimension of chain for analysis

                        # Compute the autocorrelation time (tau) so far
                        # Using tol=0 means that we'll always get an estimate even
                        # if it isn't trustworthy
                        tau = autocorr_new(chain)
                        autocorr[counter] = np.mean(tau)
                        # autocorr[counter] = autocorr_new(chain)

                        # Check convergence by checking if the chain is longer than
                        # 100 times the estimated autocorrelation time
                        # and if this estimate changed by less
                        # than 1%. This will define convergence.
                        converged = np.all(tau * 100 < sampler.iteration)
                        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                        if converged:
                            print("converged")
                            break
                        old_tau = tau

                        autocorr = [0 if math.isnan(
                            x) else x for x in autocorr]
                        autocorr_list[dimension].extend(autocorr)

                    flat_list = [item for sublist in autocorr_list for item in sublist]

                    #chain is organized as chain[walker,step,parameter]
                    chain = np.array(sampler.chain[:, :, :])
                    for i in range(0, nWalkers):
                        for j in range(0, nBreak):
                            store = []
                            for k in range(0, nDim):
                                store.append(chain[i][j][k])
                            db.insert_chain_data(db_params=config_file, table=obs["source"], chain=store)

                    sampler.reset()
                '''
                # Read the database to retrieve the data
                chains = db.get_chains(
                    db_params=config_file, 
                    table=obs["source"], 
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
                file_out = "{0}/radex-plots/new/corner_{1}.jpeg".format(DIREC, obs["source"])
                file_out_walk = "{0}/radex-plots/new/walks/walk_{1}.jpeg".format(DIREC, obs["source"])
                c = ChainConsumer() 
                c.add_chain(chain, parameters=[param1,param2,param3,param4], walkers=nWalkers)
                c.configure(color_params="posterior", cloud=True, usetex=True, summary=False) 
                fig = c.plotter.plot(filename=file_out, display=False)
                # fig.set_size_inches(6 + fig.get_size_inches())
                # summary = c.analysis.get_summary()

                fig_walks = c.plotter.plot_walks(filename=file_out_walk, display=False, plot_posterior=True)
                
