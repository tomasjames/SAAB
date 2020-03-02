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

from config import config_file, connect
from inference import *

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
        'source_flux_Jy': [14.3/1e3, 18.5/1e3], # Flux of species in Jy (original in mJy)
        'source_flux_error_Jy': [2.9/1e3, 3.6/1e3] # Error for each flux in Jy (original in mJy)
    }
    
    # Define molecular data
    sio_data = {
        'T': [75, 150., 1000],
        'g_u': 15.0,
        'A_ul': 10**(-2.83461),
        'E_u': 58.34783,
        'Z': [72.327, 144.344, 962.5698]
    }

    so_data = {
        'T': [75., 150., 1000.],
        'g_u': 17.0,
        'A_ul': 10**(-1.94660),
        'E_u': 62.14451,
        'Z': [197.515, 414.501, 1163.0700] # Scaled to upper temp with lower limit from Splatalogue (https://www.cv.nrao.edu/php/splat/species_metadata_displayer.php?species_id=20)
    }

    # Connecting to the database using 'connect()' method
    config = config_file(db_init_filename='database.ini', section='postgresql')
    # db = connect(config=config_file)

    # Define the commands necessary to create table in database
    commands = (
        """
        CREATE TABLE chain_storage (
            id SERIAL PRIMARY KEY,
            walker REAL NOT NULL,
            chain REAL NOT NULL,
            N_SIO REAL NOT NULL,
            N_SO REAL NOT NULL
        )
        """,
    )

    # Create the table
    create_table(config_file, commands)

    '''
    # Determine the estimated column densities based on the temperature (and a number of 
    # other assumptions)
    physical_conditions = param_constraints(observed_data, sio_data, so_data)
    '''   

    continueFlag = False
    nWalkers = 8
    nDim = 4 # Number of dimensions within the parameters
    nSteps = int(1e5) 
    
    prediction_file = open("{0}/radex-output/predictions.csv".format(DIREC),"w")
    
    sampler = mc.EnsembleSampler(nWalkers, nDim, ln_likelihood, args=[observed_data], pool=Pool())
    
    pos = []
    f = []

    if not os.path.exists('{0}/chains/'.format(DIREC)):
        os.makedirs('{0}/chains/'.format(DIREC))
    if not continueFlag:
        for i in range(nWalkers):
            dens = random.uniform(2, 6)
            N_sio = random.uniform(12, 14)
            N_so = random.uniform(11, 13)
            T = random.uniform(50, 500)
            pos.append([T, dens, N_sio, N_so])
            f.append(open("{0}/chains/mcmc_chain{1}.csv".format(DIREC,i+1),"w"))
    else:
        for i in range(nWalkers):
            temp=np.loadtxt("{0}/chains/mcmc_chain{0}.csv".format(DIREC,i+1))
            pos.append(list(temp[-1,:]))
            f.append(open("{0}/chains/mcmc_chain{1}.csv".format(DIREC,i+1),"a"))

    #Don't want something to go wrong mid chain and we lose everything
    #So do 10% of intended chain at a time and write out, loop.
    nBreak=int(nSteps/10)
    for counter in range(0,10):
        sampler.reset() #lose the written chain
        try: 
            pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 
        except ValueError:
            pos.pop() # Removes errorneous combination
            # Reselect values
            dens = random.uniform(2, 6)
            N_sio = random.uniform(12, 14)
            N_so = random.uniform(11, 13)
            T = random.uniform(10, 500)
            pos.append([T, dens, N_sio, N_so])
            pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 
        delete_radex_io(observed_data["species"])
        chain = np.array(sampler.chain[:,:,:]) #get chain for writing
        chain = chain.astype(float)
        #chain is organized as chain[walker,step,parameter]
        for i in range(0, nWalkers):
            for j in range(0, nBreak):
                outString = ""
                for k in range(0, nDim):
                    outString += "{0:.5f}\t".format(chain[i][j][k])
                f[i].write(outString+"\n")
        print("{0:.0f}%".format((counter+1)*10))

        sampler.reset()

    # Close the files (as they do not close automatically and will
    # not save their contents until they are closed)
    for i in range(0, nWalkers):
        f[i].close()
    
    # Open each file containing a walker's chain
    fnames=glob.glob('{0}/chains/mcmc_chain*.csv'.format(DIREC))
    print(len(fnames))
    n_walkers=len(fnames)

    # Determine the length of the first chain (assuming all chains are the same length)
    chain_length = len(np.loadtxt(fnames[0]))

    # Throw away the first 20% or so of samples so as to avoid considering initial burn-in period
    # And concatenate the chain so as chain is contiguous array
    # TODO Add Geweke test to ensure convergence has occured outside of burn-in period
    arrays = [np.loadtxt(f)[int(chain_length*0.2):] for f in fnames] 
    chain = np.concatenate(arrays)

    #Name params for chainconsumer (i.e. axis labels)
    param1 = "T / K"
    param2 = "log(n$_{H}$) / cm$^{-3}$"
    param3 = "log(N$_{SiO}$) / cm$^{-2}$"
    param4 = "log(N$_{SO}$) / cm$^{-2}$"

    #Chain consumer plots posterior distributions and calculates
    #maximum likelihood statistics as well as Geweke test
    file_out = "{0}/radex-plots/corner.jpeg".format(DIREC)
    c = ChainConsumer() 
    c.add_chain(chain, parameters=[param1,param2,param3,param4], walkers=n_walkers)
    c.configure(color_params="posterior", cloud=True, usetex=True, summary=False) 
    fig = c.plotter.plot(filename=file_out, display=False)
    # fig.set_size_inches(6 + fig.get_size_inches())
    # summary = c.analysis.get_summary()
    