import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# This is so that the code can execute from a directory
# above the UCLCHEM directory
sys.path.insert(1, "{0}/UCLCHEM/".format(os.getcwd()))
sys.path.insert(1, "{0}/UCLCHEM/scripts/".format(os.getcwd()))

import numpy as np

import databasefunctions as db
import inference
import uclchem
import plotfunctions


def parse_data(data, db_pool, db_bestfit_pool):
    """
    Parses the data from the data list in to a 
    dict list for access by the inference script. Also
    undertakes some unit conversion to ensure quantities
    are comparable with Radex data in subsequent steps.

    Arguments:
        data [list] -- a list containing all of the data
            to be parsed 
        db_pool [object] -- a pool object to store data; this
            function sets up the necessary tables
        db_bestfit_pool [object] -- a pool object to store bestfit 
            data; this function sets up the necessary tables

    Returns:
        observed_data [list] -- a dict list containing the observed
            data, formatted by observation source and molecule.
    """    

    # Define dictionary to store data
    data_storage = reset_data_dict()

    # And a list to store the data dictionaries
    observed_data = []

    # Declare a variable to help track the concurrent sources
    source_indx = 0

    # Extract the columns from the data and insert in to data list
    for indx, entry in enumerate(data[1:]):  # data[0] is the header row
        source_name = entry[0]  # The source name
        sample_size = entry[3]  #  The sample size (in beams)
        species = entry[4][0:entry[4].find(" ")].upper() # The molecule of interest
        transitions = entry[4][entry[4].find(" ")+1:].replace("–", "--").replace(
            "(", "_").replace(")", "").replace(",", "_")  # The transition of the molecule
        source_flux_dens_Jy = (float(entry[5][0:entry[5].find("±")])/1e3) #  The observed flux density (Jy)
        source_flux_dens_error_Jy = (float(entry[5][entry[5].find("±")+1:])/1e3)  # The observed flux error (Jy)
        linewidths = abs(float(entry[7])*1e3)  # The linewidth (m/s)

        if species == "SIO":
            transition_freq = 303.92696000*1e9
        elif species == "SO":
            transition_freq = 304.07791400*1e9
        elif species == "OCS":
            transition_freq = 303.99326100*1e9
        elif species == "H2CS":
            transition_freq = 304.30774200*1e9
        else:
            transition_freq = np.nan

        # Convert flux from Jy to Rayleigh-Jeans estimate in K
        rj_equiv = inference.rj_flux(
            source_flux_dens_Jy,
            transition_freq,
            linewidths/1e3
        )

        rj_equiv_error = inference.rj_flux(
            source_flux_dens_error_Jy,
            transition_freq,
            linewidths/1e3
        )

        # Check to see whether this is the first § entry
        # Or whether the source already exists within the conditioned dictionary
        if indx == 0 or observed_data[len(observed_data)-1]['source'] != source_name:
            source_indx += 1
            data_storage['source'] = source_name
            data_storage['sample_size'] = sample_size
            data_storage['species'].append(species)
            data_storage['transitions'].append(transitions)
            data_storage['source_flux_dens_Jy'].append(source_flux_dens_Jy)
            data_storage['source_flux_dens_error_Jy'].append(source_flux_dens_error_Jy)
            data_storage['linewidths'].append(abs(linewidths)) # abs catches any -ve linewidths
            data_storage['transition_freqs'].append(transition_freq)
            data_storage["source_rj_flux"].append(rj_equiv["rj_flux"])
            data_storage["source_rj_flux_error"].append(rj_equiv_error["rj_flux"])

            # Store the data
            observed_data.append(data_storage)
            data_storage = reset_data_dict()  # Reset the data dict
        else:
            # Append the data to the pre-existing entry in the dict-list
            observed_data[source_indx-1]['species'].append(species)
            observed_data[source_indx-1]['transitions'].append(transitions)
            observed_data[source_indx-1]['source_flux_dens_Jy'].append(source_flux_dens_Jy)
            observed_data[source_indx-1]['source_flux_dens_error_Jy'].append(source_flux_dens_error_Jy)
            observed_data[source_indx-1]['linewidths'].append(linewidths)
            observed_data[source_indx-1]["source_rj_flux"].append(rj_equiv["rj_flux"])
            observed_data[source_indx-1]["source_rj_flux_error"].append(rj_equiv_error["rj_flux"])
            observed_data[source_indx-1]['transition_freqs'].append(transition_freq)

    return observed_data


def filter_data(data, species):
    """
    A function that filters a given dictionary (data) to 
    only contain species including within the species
    list (species).

    Arguments:
        data {dict} -- A dictionary containing data
                        from parse_data
        species {list} -- A list containing any and all
                            species of interest

    Returns:
        {dict} -- A dictionary of the same format as that 
                    supplied but with irrelevant species removed
    """    
    filtered_data = [] # List to hold the resulting data
    for source in data: # Loop through the data
        diffs = list(set(source["species"]) - set(species)) # Find the differences between species in data and species of interest
        for unique in diffs: # Remove those unique elements found above
            indx = source["species"].index(unique)
            source["species"].pop(indx)
            source["transitions"].pop(indx)
            source["transition_freqs"].pop(indx)
            source["source_flux_dens_Jy"].pop(indx)
            source["source_flux_dens_error_Jy"].pop(indx)
            source["linewidths"].pop(indx)
            source["source_rj_flux"].pop(indx)
            source["source_rj_flux_error"].pop(indx)
        filtered_data.append(source)
    return filtered_data


def run_uclchem(vs, n, t_evol, species, DIREC):

    # Define the table name
    file_name = "phase1-n{0:.6E}".format(n)

    # Set r_out in order to set the extinction correctly
    # within UCLCHEM
    if n <= 1e2:
        r_out = 5.3
    elif 1e2 < n <= 1e3:
        r_out = 2.6
    elif 1e3 < n <= 1e4:
        r_out = 0.5
    elif 1e4 < n <= 1e5:
        r_out = 0.25
    else:
        r_out = 0.05

    # Check if phase 1 exists (as n is the only important factor here)
    try:
        # db.does_table_exist(db_params=config_file, table=file_name)
        open("{0}/UCLCHEM/output/start/{1}.dat".format(DIREC, file_name))
    except FileNotFoundError:
        uclchem.general(
            {
                "initialDens": 1e2,
                "finalDens": n,
                "rout": r_out,
                "phase": 1,
                "readAbunds": 0,
                "abundFile": "{0}/UCLCHEM/output/start/{1}.dat".format(DIREC, file_name),
                "outputFile": "{0}/UCLCHEM/output/start/full_output_{1}.dat".format(DIREC, file_name)
            },
        species)
    finally:
        #Run UCLCHEM
        uclchem.general(
            {
                "initialDens": 2*n,
                "finalDens": n,
                "finalTime": t_evol,
                "vs": vs,
                "rout": r_out,
                "switch": 0,
                "phase": 2,
                "readAbunds": 1,
                "desorb": 1,
                "abundFile": "{0}/UCLCHEM/output/start/{1}.dat".format(DIREC, file_name),
                "outputFile": "{0}/UCLCHEM/output/data/v{1:.6E}n{2:.6E}.dat".format(DIREC, vs, n)
            },
            ["SIO", "SO"])
    
        print("UCLCHEM run complete")

        times, dens, temp, abundances = plotfunctions.read_uclchem(
            "{0}/UCLCHEM/output/data/v{1:.6E}n{2:.6E}.dat".format(DIREC, vs, n), species)
    
        # Determine the H column density through the shock
        coldens = [r_out*(3e18)*n_i for n_i in dens]

        return {"times": times,
                "dens": dens,
                "temp": temp,
                "abundances": abundances,
                "H_coldens": coldens}


'''
def read_shock_data(filename):
    a = open(filename).read()
    a = a.split('\n')
    
    time, dens, temp, vel, av, N = ([] for i in range(6))
    
    with open(filename) as file:
        for line in file:
            bits = line.split()
            #find time line
            if 'age' in bits:
                time.append(float(bits[-2].replace('D', 'E')))
            #read another line for dens
            if 'density' in bits:
                densi = float(bits[-2].replace('D', 'E'))
                if densi == 0.0:
                    densi = 1e-10
                dens.append(densi)
            if "temp" in bits:
                tempi = float(bits[-2].replace('D', 'E'))
                temp.append(tempi)
            if "velocity" in bits:
                vel.append(float(bits[-2].replace('D', 'E')))
            if "extinction" in bits:
                av.append(float(bits[-2].replace('D', 'E')))
            if "column" in bits:
                N.append(float(bits[-2].replace('D', 'E')))

    return time, dens, temp, vel, av, N
'''


def resolved_quantity(density, measure, t):
    
    for indx in range(1, len(density)):
        dt = t[indx] - t[indx-1] 
        numerator =+ density[indx]*measure[indx]*dt
        denominator =+ density[indx]*dt

    quantity = numerator/denominator
    return quantity


'''
def time_distance_transform(vel, times):
    # Transform time in to distance (with 0 based array to account for the fact that
    # distance begins from 0)
    distance = [0]

    for i in range(1, len(times)):
        distance.append((vel[i]*1e5)*(times[i] - times[i-1]) * (60*60*24*365) + distance[i-1])

    return distance
'''


def reset_data_dict():
    data = {
        'source': "",  # The source name
        'sample_size': 0,  # The sample size in beams
        'species': [],  # Species of interest
        'transitions': [],  #  Transitions of interest
        'transition_freqs': [],  # Transition frequencies according to Splatalogue in GHz
        'linewidths': [],  # Linewidths for the transitions in km/s
        'source_flux_dens_Jy': [],  # Flux of species in Jy (original in mJy)
        #  Error for each flux in Jy (original in mJy)
        'source_flux_dens_error_Jy': [],
        'source_rj_flux': [], # The Rayleight Jeans flux in K
        'source_rj_flux_error': [] # The error on the above flux in K
    }
    return data
