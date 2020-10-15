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

if __name__ == '__main__':

    # Define constants
    DIREC = os.getcwd()

    # Define emcee specific parameters
    nWalkers = 100  # Number of random walkers to sample parameter space
    nSteps = int(1e2)  # Number of steps per walker

    # Declare the species that we're interested in
    relevant_species = ["SIO", "SO", "H2CS", "OCS"]

    # Define the datafile
    datafile = "{0}/data/tabula-sio_intratio.csv".format(DIREC)

    radex_config_file = config.config_file(
        db_init_filename='database_archi.ini', section='radex_fit_results')
    bestfit_config_file = config.config_file(
        db_init_filename='database_archi.ini', section='radex_bestfit_conditions')

    # Set up the connection pools
    db_radex_pool = db.dbpool(radex_config_file)
    db_bestfit_pool = db.dbpool(bestfit_config_file)

    # Read in the whole data file containing sources and source flux
    with open(datafile, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()

    # Parse the data to a dict list
    observed_data = workerfunctions.parse_data(
        data, db_radex_pool, db_bestfit_pool)

    # Filter the observed data to contain only those species that we can use
    # (normally limited by those with Radex data)
    filtered_data = workerfunctions.filter_data(
        observed_data, relevant_species)

    # Set up chain
    c = ChainConsumer()

    # Begin by looping through all of the observed sources
    # and start by creating a database for each entry
    for obs in filtered_data:
        if (len(obs["species"]) >= 2 and "SIO" in obs["species"]) or \
                (len(obs["species"]) >= 2 and "SO" in obs["species"]) or \
            (len(obs["species"]) >= 2 and "OCS" in obs["species"]) or \
                (len(obs["species"]) >= 2 and "H2CS" in obs["species"]):

            print("Getting data from database")

            # Define the column names for querying the database
            column_names = ["temp", "dens"] + \
                ["column_density_{0}".format(spec) for spec in obs["species"]]

            # Name params for chainconsumer (i.e. axis labels)
            plot_params = ["T [K]", "log(n$_{H}$) [cm$^{-3}$]"] + [
                "log(N$_{{{0}}}$)[cm$^ {{-2}}$]".format(spec) for spec in obs["species"]]

            if len(plot_params) == 6:
                params = plot_params

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
            c.add_chain(chain, parameters=plot_params, walkers=nWalkers, name=obs["source"])
    
    # Get summary
    summary = c.analysis.get_summary(parameters=params)

    # Extract mean values
    temps = np.array([x[params[0]][1] for x in summary])
    dens = np.array([x[params[1]][1] for x in summary])
    N_SiO = np.array([x[params[2]][1] for x in summary])
    N_SO = np.array([x[params[3]][1] for x in summary])
    try:
        N_H2CS = np.array([x[params[4]][1] for x in summary])
    except IndexError:
        print("whoopsie")
    try:
        N_OCS = np.array([x[params[5]][1] for x in summary])
    except IndexError:
        print("whoopsie")

    # Calculate rms
    rms_temp = np.sqrt(np.mean(temps**2))
    rms_dens = np.sqrt(np.mean(dens**2))
    rms_N_SiO = np.sqrt(np.mean(N_SiO**2))
    rms_N_SO = np.sqrt(np.mean(N_SO**2))
    rms_N_H2CS = np.sqrt(np.mean(N_H2CS**2))
    rms_N_OCS = np.sqrt(np.mean(N_OCS**2))

    print("RMS temp: {0}".format(rms_temp))
    print("RMS dens: {0}".format(rms_dens))
    print("RMS N_SiO: {0}".format(rms_N_SiO))
    print("RMS N_SO: {0}".format(rms_N_SO))
    print("RMS N_H2CS: {0}".format(rms_N_H2CS))
    print("RMS N_OCS: {0}".format(rms_N_OCS))

    table = c.analysis.get_latex_table(
        caption="Radex bestfit ranges for each source", label="tab:model-summaries")

