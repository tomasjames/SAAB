import io
import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# This is so that the code can execute from a directory
# above the UCLCHEM directory
sys.path.insert(1, "{0}/UCLCHEM/".format(os.getcwd()))
import uclchem
sys.path.insert(1, "{0}/UCLCHEM/scripts/".format(os.getcwd()))
import plotfunctions

import matplotlib.pyplot as plt
import numpy as np

import databasefunctions as db
import inference

# For the redirect stdout and stederr
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper


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
        print(entry)
        source_name = entry[0]  # The source name
        num_beams = float(entry[3])  #  The number of beams in the observation
        species = entry[4][0:entry[4].find(" ")].upper() # The molecule of interest
        if species == "SO":
            transitions = "7_8--6_7"
        else:
            transitions = entry[4][entry[4].find(" ")+1:].replace("–", "--").replace(
                "(", "_").replace(")", "").replace(",", "_")  # The transition of the molecule
        source_flux_dens_Jy = (float(entry[5][0:entry[5].find("±")])/1e3) #  The observed flux density (Jy)
        source_flux_dens_error_Jy = (float(entry[5][entry[5].find("±")+1:])/1e3)  # The observed flux error (Jy)
        linewidths = abs(float(entry[7])*1e3)  # The linewidth (m/s)

        print("source_flux_dens_Jy={0}".format(source_flux_dens_Jy))

        if species == "SIO":
            transition_freq = 303.92696000*1e9
        elif species == "SO":
            transition_freq = 304.07791400*1e9
        elif species == "OCS":
            transition_freq = 303.99326100*1e9
        elif species == "H2CS":
            transition_freq = 304.30774200*1e9
        elif species == "CH3OH":
            transition_freq = 304.20832400*1e9
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
            data_storage['num_beams'] = num_beams
            data_storage['species'].append(species)
            data_storage['transitions'].append(transitions)
            data_storage['source_flux_dens_Jy'].append(source_flux_dens_Jy)
            data_storage['source_flux_dens_error_Jy'].append(source_flux_dens_error_Jy)
            data_storage['linewidths'].append(abs(linewidths)) # abs catches any -ve linewidths
            data_storage['transition_freqs'].append(transition_freq)
            data_storage["source_rj_flux"].append(round(rj_equiv["rj_flux"], 3))
            data_storage["source_rj_flux_error"].append(round(rj_equiv_error["rj_flux"], 6))

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
            observed_data[source_indx-1]["source_rj_flux"].append(round(rj_equiv["rj_flux"], 3))
            observed_data[source_indx-1]["source_rj_flux_error"].append(round(rj_equiv_error["rj_flux"], 6))
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
        if len(source["species"]) < 2:
            print(source)
            continue
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


def get_r_out(n):
    
    # Set r_out in order to set the extinction correctly
    # within UCLCHEM
    if n <= 1e2:
        r_out = 5.3
    elif 1e2 < n <= 1e3:
        r_out = 1.0
    elif 1e3 < n <= 1e4:
        r_out = 0.35
    elif 1e4 < n <= 1e5:
        r_out = 0.2
    else:
        r_out = 0.1
    
    return r_out

def run_uclchem(phase1, phase2, species):

    print("\n Running UCLCHEM..")
    # Check if phase 1 exists (as n is the only important factor here)
    print("\t Running phase 1...")
    uclchem.wrap.run_model_to_file(phase1, species)
    print("\t Phase 1 complete \n")
    
    print("\t Running phase 2...")
    uclchem.wrap.run_model_to_file(phase2, species)
    print("\t Phase 2 complete \n")

    print("UCLCHEM run complete \n Reading UCLCHEM output...")
    data = plotfunctions.read_uclchem("{0}".format(phase2["outputFile"])) 
    times, dens, temp, abundances = list(data["Time"]), list(data["Density"]), list(data["gasTemp"]), []   

    for spec in species.split(): # split() required as species is a string
        print("spec={0}".format(spec))
        print("data[spec]={0}".format(data[spec]))
        abundances.append(data[spec])

    # Determine the H column density through the shock
    coldens = [phase2["rout"]*(3e18)*n_i for n_i in dens]

    return {
        "times": times,
        "dens": dens,
        "temp": temp,
        "abundances": abundances,
        "H_coldens": coldens
    }


def plot_uclchem(model, species, plotfile):

    # Determine the H2 column density
    H_coldens = 0.5*abund(nh2, dstep)*density(dstep)*(cloudSize/real(points))

    # Set up the plot    
    if species[-1][0] == "#":
        figs = 3
    else:
        figs = 2
    
    fig, ax = plt.subplots(figs, figsize=(8, 8))

    # Twin the last axis and the first
    ax_first_twin = ax[0].twinx()
    ax_final_twin = ax[-1].twinx()

    # Plot abundances
    for spec_indx, spec_name in enumerate(species):
        if spec_name[0] != "#":
            ax[0].loglog(model["times"], model["abundances"][spec_indx], label=spec_name)
            ax_first_twin.loglog(model["times"], [a*b for (a,b) in zip(H_coldens, model["abundances"][spec_indx])], alpha=0.2, linestyle=":")
        if spec_name[0] == "#":
            ax[1].loglog(model["times"], model["abundances"][spec_indx], label=spec_name)

    # Plot temp and dens
    ax[-1].loglog(model["times"], model["dens"], linestyle="--", color="r", label="n$_{H}$")
    ax_final_twin.loglog(model["times"], model["temp"], linestyle=":", color="g", label="T")

    # Remove ticks from axis 0 and 1
    ax[0].set_xticks([])
    ax[1].set_xticks([])

    # Plot legends
    for axis in ax[:-1]:
        axis.legend(loc='best', fontsize='small')
    # ax_first_twin.legend(loc='best', fontsize='small')
    # ax_final_twin.legend(loc='best', fontsize='small')
    
    # Set labels
    ax[0].set_ylabel("X$_{Species}$")
    ax_first_twin.set_ylabel("N$_{Species}$ [cm$^{-2}$]")
    if figs == 3:
        ax[1].set_ylabel("X$_{Species}$")
    ax[-1].set_ylabel("n$_{H}$ [cm$^{-3}$]", color="r")
    ax_final_twin.set_ylabel("T [K]", color="g")
    ax[-1].set_xlabel('t [yrs]')

    # Compress the plot
    plt.tight_layout()

    # Save the files
    fig.savefig(plotfile)

    # Close plot
    # plt.close()

    return 


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


def resolved_quantity(density, measure, t):
    
    for indx in range(1, len(density)):
        dt = t[indx] - t[indx-1] 
        numerator =+ density[indx]*measure[indx]*dt
        denominator =+ density[indx]*dt

    quantity = numerator/denominator
    return quantity


def time_distance_transform(vel, times):
    # Transform time in to distance (with 0 based array to account for the fact that
    # distance begins from 0)
    distance = [0]

    for i in range(1, len(times)):
        distance.append((vel[i]*1e5)*(times[i] - times[i-1]) * (60*60*24*365) + distance[i-1])

    return distance


def reset_data_dict():
    data = {
        'source': "",  # The source name
        'num_beams': 0,  # The sample size in beams
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
