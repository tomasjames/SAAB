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
import correlation
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
        dens = random.uniform(4, 8)
        params['dens'] = dens
    if params['N_sio']:
        N_sio = random.uniform(12, 15)
        params['N_sio'] = N_sio
    if params['N_so']:
        N_so = random.uniform(12, 15)
        params['N_so'] = N_so
    if params['N_ch3oh']:
        N_ch3oh = random.uniform(13, 17)
        params['N_ch3oh'] = N_ch3oh
    if params['N_ocs']:
        N_ocs = random.uniform(13, 16)
        params['N_ocs'] = N_ocs
    if params['N_h2cs']:
        N_h2cs = random.uniform(13, 16)
        params['N_h2cs'] = N_h2cs

    return params


# Define constants
DIREC = os.getcwd()
RADEX_PATH = "{0}/../Radex".format(DIREC)

plot_results = True

if __name__ == '__main__':

    # Define emcee specific parameters
    nWalkers = 500  # Number of random walkers to sample parameter space
    nSteps = int(1e4)  # Number of steps per walker

    #Set up MPI Pool
    #pool = Pool(24)

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
    relevant_species = ["SIO", "SO", "CH3OH", "H2CS", "OCS"]

    # Declare the database connections
    radex_config_file = config.config_file(db_init_filename='database_archi.ini', section='radex_fit_results')
    bestfit_config_file = config.config_file(db_init_filename='database_archi.ini', section='radex_bestfit_conditions')

    # Set up the connection pools
    db_radex_pool = db.dbpool(radex_config_file)
    db_bestfit_pool = db.dbpool(bestfit_config_file)

    # Read in the whole data file containing sources and source flux
    with open(datafile, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()

    # Remove OCS from K13a
    for element in data:
        print(element)
        if element[0] == "K13a" and "OCS" in element[-5]:
            data.remove(element)

    # Parse the data to a dict list
    observed_data = workerfunctions.parse_data(data)
    
    # Filter the observed data to contain only those species that we can use
    # (normally limited by those with Radex data)
    filtered_data = workerfunctions.filter_data(observed_data, relevant_species)

    # Begin by looping through all of the observed sources 
    # and start by creating a database for each entry
    physical_conditions = []
    for obs_indx, obs in enumerate(filtered_data):
        if (len(obs["species"]) >= 2 and "SIO" in obs["species"]) or \
                (len(obs["species"]) >= 2 and "SO" in obs["species"]) or \
                    (len(obs["species"]) >= 2 and "CH3OH" in obs["species"]) or \
                        (len(obs["species"]) >= 2 and "OCS" in obs["species"]) or \
                            (len(obs["species"]) >= 2 and "H2CS" in obs["species"]):
           
            #print("source={0}".format(obs["source"])) 
            #'''
            #start_index = next((index for (index, d) in enumerate(filtered_data) if d["source"] == "L20"), None)
            #if obs_indx < start_index:
            #    continue
            if obs["source"] == "K13a":
                print("Source is {}".format(obs["source"]))
            else:
                #print("Source is {}".format(obs["source"]))
                continue
            '''     
            #Checks to see whether the tables exists; if so, delete it
            if db.does_table_exist(db_pool=db_radex_pool, table=obs["source"]):
                #continue
                db.drop_table(db_pool=db_radex_pool, table=obs["source"])
            if db.does_table_exist(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"])):
                db.drop_table(db_pool=db_bestfit_pool, table="{0}_bestfit_conditions".format(obs["source"]))
                #continue 

            # Store the column density string
            all_column_str = "" # Empty string to hold full string once complete
            for spec_indx, spec in enumerate(obs["species"]):
                if spec == "SIO" or spec == "SO" or spec == "CH3OH" or spec == "H2CS" or spec == "OCS":
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
                    temp INTEGER NOT NULL,
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
                    temp INTEGER NOT NULL,
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
            sio_flag, so_flag, ch3oh_flag, h2cs_flag, ocs_flag = False, False, False, False, False

            pos = []

            # Refine number of walkers
            #nWalkers = (len(obs["species"]) + 2)*2

            # Determine which molecules are present and alter flags
            if "SIO" in obs["species"]:
                sio_flag = True
            if "SO" in obs["species"]:
                so_flag = True
            if "CH3OH" in obs["species"]:
                ch3oh_flag = True
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
                    'N_ch3oh': ch3oh_flag,
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
            # print("nDim={0}".format(nDim))

            # Define the column names for saving to the database
            column_names = ["temp", "dens"] + ["column_density_{0}".format(spec) for spec in obs["species"]]

            # Set up the backend
            # Don't forget to clear it in case the file already exists
            #filename = "{0}.h5".format(obs["source"])
            #backend = mc.backends.HDFBackend(filename)
            #backend.reset(nWalkers, nDim)

            # Run the sampler
            sampler = mc.EnsembleSampler(nWalkers, nDim, inference.ln_likelihood_radex, 
                args=(obs, bestfit_config_file, DIREC, RADEX_PATH), pool=Pool(24))

            # Split the chain in to 100 chunks, each 1% of the total size and write out
            nBreak=int(nSteps/100)
            for counter in range(nBreak):
                sampler.reset() # Reset the chain

                print("Running mcmc")

                pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 

                print("MCMC complete on counter {}".format(counter))                
                print("Moving to file deletion")

                for species in obs["species"]:
                    # Delete the Radex and UCLCHEM input and output files
                    inference.delete_radex_io(species, DIREC)

                #chain is organized as chain[walker, step, parameter(s)]
                #chain = np.array(sampler.chain[:, :, :])
                chain = np.array(sampler.chain)

                for i in range(0, nWalkers):
                    for j in range(0, nBreak):
                        print("Inserting chain[{0}][{1}]".format(i, j))
                        db.insert_radex_chain_data(db_pool=db_radex_pool, table=obs["source"], chain=chain[i][j], column_names=column_names)                   

                if counter == 0:
                    print("counter == 0")
                    full_chain = np.vstack(chain)
                else:
                    print("Stacking chains")
                    full_chain = np.concatenate((full_chain, np.vstack(chain)))

                print("Checking convergence")
                # Checks for convergence using ChainConsumer
                c = ChainConsumer()
                c.add_chain(full_chain, parameters=column_names, walkers=nWalkers)

                # Convergence tests
                gelman_rubin = c.diagnostic.gelman_rubin(threshold=0.15)
                if gelman_rubin:
                    print("Chains have converged")
                    break
                else:
                    print("Chains have yet to converge")    

            '''
            if plot_results:
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
                chain = np.array(chains)

                # Ensure chain is truncated to maximum chain length
                #if obs["source"] == "G2":
                #    chain = chain[:int(nWalkers*nSteps/10)]
                #    continue

                # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
                #chain = chain[int(nWalkers*nSteps*0.1):]
                chain = np.array(chains[int(chain_length*0.2):])

                # Name params for chainconsumer (i.e. axis labels)
                plot_params = ["T [K]", "log(n$_{H}$) [cm$^{-3}$]"] + [
                    "log(N$_{{{0}}}$)[cm$^ {{-2}}$]".format(spec) for spec in obs["species"]]

                # Chain consumer plots posterior distributions and calculates
                # maximum likelihood statistics as well as Geweke test
                file_out = "{0}/radex-plots/new/corner_{1}.pdf".format(DIREC, obs["source"])
                file_out_walk = "{0}/radex-plots/new/walks/walk_{1}.pdf".format(DIREC, obs["source"])
                c = ChainConsumer() 
                c.add_chain(chain, parameters=plot_params, walkers=nWalkers)

                # Convergence tests
                gelman_rubin = c.diagnostic.gelman_rubin(threshold=0.15)
                if gelman_rubin:
                    print("Chains have converged")
                else:
                    print("Chains have yet to converge")

                c.configure(color_params="posterior", usetex=True, summary=False, bins=0.3, cloud=False, spacing=2.0, sigmas=np.linspace(0, 2, 3), plot_contour=True, bar_shade=[False]*len(plot_params))
                fig = c.plotter.plot(filename=file_out, display=False,) #extents=[(60, 1000), (4, 7), (13, 15), (13, 15), (14, 17), (14, 15), (14, 15)])

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
