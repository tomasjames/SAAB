# -*- coding: utf-8 -*-

import csv
import os
from multiprocessing import Pool
import psycopg2
import sys
import subprocess as sp

from pdf2image import convert_from_path

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
        T_lower = 60.0  #  Lower limit set by Eu for SiO and SO transitions 
        if params["N_h2cs"]:
            T_lower = 86.0 # Limit from Splatalogue for 9(1,9)--8(1,8)
        if params["N_ocs"]:
            T_lower = 190.0  # Limit from Splatalogue for 25--24
        T = random.uniform(T_lower, 1000)
        params['temp'] = T
    if params['dens']:
        dens = random.uniform(3, 7)
        params['dens'] = dens
    if params['N_sio']:
        N_sio = random.uniform(12, 15)
        params['N_sio'] = N_sio
    if params['N_so']:
        N_so = random.uniform(12, 15)
        params['N_so'] = N_so
    if params['N_ocs']:
        N_ocs = random.uniform(12, 16)
        params['N_ocs'] = N_ocs
    if params['N_h2cs']:
        N_h2cs = random.uniform(12, 16)
        params['N_h2cs'] = N_h2cs

    return params


# Define constants
DIREC = os.getcwd()
RADEX_PATH = "{0}/../Radex".format(DIREC)
#RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"

print_results = True

if __name__ == '__main__':

    # Define emcee specific parameters
    nWalkers = 100  # Number of random walkers to sample parameter space
    nSteps = int(1e2)  # Number of steps per walker

    #Set up MPI Pool
    pool = Pool(8)

    # Define the datafile
    datafile = "{0}/data/tabula-sio_intratio.csv".format(DIREC)

    # Define molecular data (only required for column density estimations)
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
    relevant_species = ["SIO", "SO", "H2CS", "OCS"]

    # Declare the database connections
    radex_config_file = config.config_file(db_init_filename='database_archi.ini', section='radex_fit_results')
    bestfit_config_file = config.config_file(db_init_filename='database_archi.ini', section='bestfit_conditions')

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
    filtered_data = workerfunctions.filter_data(observed_data, relevant_species)
    
    # Begin by looping through all of the observed sources 
    # and start by creating a database for each entry
    physical_conditions = []
    for obs in filtered_data:
        if (len(obs["species"]) >= 2 and "SIO" in obs["species"]) or \
                (len(obs["species"]) >= 2 and "SO" in obs["species"]) or \
                    (len(obs["species"]) >= 2 and "OCS" in obs["species"]) or \
                        (len(obs["species"]) >= 2 and "H2CS" in obs["species"]):
           
            print("source={0}".format(obs["source"])) 
            '''
            # Checks to see whether the tables exists; if so, delete it
            if db.does_table_exist(db_pool=db_radex_pool, table=obs["source"]):
                continue
                #db.drop_table(db_pool=db_radex_pool, table=obs["source"])
            if db.does_table_exist(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"])):
                #db.drop_table(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"]))
                continue 
            # Store the column density string
            all_column_str = "" # Empty string to hold full string once complete
            for spec_indx, spec in enumerate(obs["species"]):
                if spec == "SIO" or spec == "SO" or spec == "OCS" or spec == "H2CS":
                    column_str = "column_density_{0} REAL NOT NULL, ".format(spec)
                    
                    # Remove the comma on the final column density
                    if spec_indx == (len(obs["species"])-1): 
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
            
            # Empty flags to signify whether molecules are present
            # Assune False by default
            sio_flag, so_flag, h2cs_flag, ocs_flag = False, False, False, False

            pos = []

            # Determine which molecules are present and alter flags
            if "SIO" in obs["species"]:
                sio_flag = True
            if "SO" in obs["species"]:
                so_flag = True
            if "H2CS" in obs["species"]:
                h2cs_flag = True
            if "OCS" in obs["species"]:
                ocs_flag = True

            # Select the parameters
            for i in range(nWalkers):
                params = param_select({
                    'temp': True,
                    'dens': True,
                    'N_sio': sio_flag,
                    'N_so': so_flag,
                    'N_h2cs': h2cs_flag,
                    'N_ocs': ocs_flag
                })

                # Iterate through those lists to find entries that
                # are still marked as False (and not required)
                pos_list = [x for x in params.values() if x != False]

                # Append the correct params to the params list
                pos.append(pos_list)

            # Determine the number of dimensions
            nDim = len(pos[0])
            print("nDim={0}".format(nDim))

            # Define the column names for saving to the database
            column_names = ["temp", "dens"] + ["column_density_{0}".format(spec) for spec in obs["species"]]

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
                        db.insert_radex_chain_data(db_pool=db_radex_pool, table=obs["source"], chain=store, column_names=column_names)

                sampler.reset()
            '''
            if print_results:
                print("Moving to plotting routine")
                print("Getting data from database")

                # Define the column names for querying the database
                column_names = ["temp", "dens"] + ["column_density_{0}".format(spec) for spec in obs["species"]]

                # Read the database to retrieve the data
                chains = db.get_chains(
                    db_pool=db_radex_pool, 
                    table=obs["source"], 
                    column_names=column_names
                )

                # Determine the length of the first chain (assuming all chains are the same length)
                try:
                    chain_length = len(chains)
                except TypeError:
                    continue

                # and catch any chains that might be 0 length
                if chain_length == 0:
                    continue

                # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
                chain = np.array(chains[int(chain_length*0.2):])

                print(obs["species"])

                # Name params for chainconsumer (i.e. axis labels)
                plot_params = ["T [K]", "log(n$_{H}$) [cm$^{-3}$]"] + [
                    "log(N$_{{{0}}}$)[cm$^ {{-2}}$]".format(spec) for spec in obs["species"]]

                # Chain consumer plots posterior distributions and calculates
                # maximum likelihood statistics as well as Geweke test
                file_out = "{0}/radex-plots/new/corner_{1}.pdf".format(DIREC, obs["source"])
                file_out_walk = "{0}/radex-plots/new/walks/walk_{1}.pdf".format(DIREC, obs["source"])
                c = ChainConsumer() 
                c.add_chain(chain, parameters=plot_params, walkers=nWalkers)
                c.configure(color_params="posterior", usetex=True, summary=False, sigmas=[0, 1, 2, 3])
                fig = c.plotter.plot(filename=file_out, display=False)

                fig_walks = c.plotter.plot_walks(filename=file_out_walk, display=False, plot_posterior=True)
                plt.close()

                # Convert both pdf to jpeg
                corners = convert_from_path('{0}'.format(file_out))
                walks = convert_from_path('{0}'.format(file_out_walk))

                for corner, walk in zip(corners, walks):
                    corner.save('{0}.jpg'.format(file_out[:-4]), 'JPEG')
                    walk.save('{0}.jpg'.format(file_out_walk[:-4]), 'JPEG')

                    os.remove(file_out)
                    os.remove(file_out_walk)
            
    pool.close()

