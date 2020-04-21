import glob
import numpy as np
import os
import subprocess as sp
import time

import config
import databasefunctions as db
import workerfunctions



def get_trial_data(params, observed_data, DIREC, RADEX_PATH):
    
    print("Getting trial data")

    # Declare dictionary to hold trial data
    trial_data = {
        'species': [],
        'N': [],
        'flux': []
    }

    # Unpack the parameters
    vs, initial_dens = params[0], params[1]

    # Determine the dissipation length and identify
    # the point that it begins (max_temp_indx)
    dlength = 12.0*3.08e18*vs/initial_dens

    # Convert to time
    t_diss = (dlength/(vs*1e5))/(60*60*24*365)

    # Run the UCLCHEM model up to the dissipation length time analogue
    print("Running UCLCHEM")
    shock_model = workerfunctions.run_uclchem(
        vs, initial_dens, t_diss, DIREC)

    # Average the quantities across the dissipation region
    # (i.e. the time that we've evolved the model for)
    T = workerfunctions.resolved_quantity(
        shock_model["dens"][1:],
        shock_model["temp"][1:],
        shock_model["times"][1:]
    )
    n = workerfunctions.resolved_quantity(
        shock_model["dens"][1:],
        shock_model["dens"][1:],
        shock_model["times"][1:]
    )

    for spec_indx, spec in enumerate(observed_data["species"]):    
        # Store the species
        trial_data['species'].append(spec)

        # Get the abundances
        abund = shock_model["abundances"][spec_indx][1:]

        # Set the column density
        N = workerfunctions.resolved_quantity(
            shock_model["dens"][1:],
            shock_model["H_coldens"][1:]*abund,
            shock_model["times"][1:]
        )

        # Get the linewidth and relevant transition
        dv = observed_data["linewidths"][spec_indx]
        transition = observed_data["transitions"][spec_indx]

        print("Writing the radex inputs")
        # Write the radex input file
        write_radex_input(spec, n, T, n, N, 
            dv, vs, initial_dens, DIREC, RADEX_PATH, f_min=290, f_max=360)

        print("Running radex")
        # Run radex
        sp.run('radex < {0}/radex-input/{1}/vs{2}n{3:1.0E}.inp &> /dev/null'.format(
            DIREC, spec, int(vs), int(initial_dens)), shell = True)  # &> pipes stdout and stderr

        output_file_path = ('{0}/radex-output/{1}/vs{2}n{3:1.0E}.out'.format(
            DIREC, spec, int(vs), int(initial_dens)))

        # This block ensures that the output file exists before attempting to read it
        while not os.path.exists(output_file_path):
            time.sleep(1)
            print("Waiting for {0}".format(output_file_path))
        if os.path.isfile(output_file_path):
            # Read the radex output
            radex_output = read_radex_output(
                spec, transition, vs, initial_dens, DIREC)
        else:
            raise ValueError("%s isn't a file!" % output_file_path)

        # Determine the flux density
        trial_data['flux'].append(radex_output["flux"])
        trial_data['N'].append(N)

    return trial_data


# Function to write the input to run Radex
def write_radex_input(spec, ns, tkin, nh2, N, dv, vs, initial_dens, DIREC, RADEX_PATH, f_min, f_max):
    tbg=2.73
    # Open the text file that constitutes Radex's input file
    infile = open('{0}/radex-input/{1}/vs{2}n{3:1.0E}.inp'.format(
        DIREC, spec, int(vs), int(initial_dens)), 'w')
    infile.write('{0}/data/{1}.dat\n'.format(RADEX_PATH,spec.lower()))  # Molecular data file
    infile.write('{0}/radex-output/{1}/vs{2}n{3:1.0E}.out\n'.format(DIREC, spec, int(vs), int(initial_dens)))  # Output file
    infile.write('{0} {1}\n'.format(f_min, f_max)) # Frequency range (0 0 is unlimited)
    infile.write('{0}\n'.format(tkin)) # Kinetic temperature (i.e. temp of gas)
    infile.write('1\n') # Number of collisional partners
    infile.write('H2\n') # Type of collisional partner
    infile.write('{0}\n'.format(nh2)) # Density of collision partner (i.e. dens of gas)
    infile.write('{0}\n'.format(tbg)) # Background temperature
    infile.write('{0}\n'.format(N)) # Column density of emitting species
    infile.write('{0}\n'.format(dv/1e3)) # Line width
    infile.write('0 \n') # Indicates Radex should exit

    infile.close()


def read_radex_output(spec, transition, vs, initial_dens, DIREC):

    radex_output = {
        "temp": 0, 
        "dens": 0, 
        "E_up": 0, 
        "wav": 0, 
        "flux": 0
    }

    with open('{0}/radex-output/{1}/vs{2}n{3:1.0E}.out'.format(
        DIREC, spec, int(vs), int(initial_dens)), 'r') as outfile:
        lines = outfile.readlines()

        # Loop through the simulation information in the header of the 
        # output file
        for line in lines[:8]:
            words = line.split()
            # Extract kinetic temperature
            if (words[1] == "T(kin)"): 
                radex_output["temp"] = float(words[-1])
            # Extract density
            if (words[1] == "Density"):
                radex_output["dens"] = float(words[-1])
        
        for line in lines[11:]:
            words = line.split()

            # Extract the RADEX data (easiest is to recognise when -- is used to 
            # define the line transition)
            if (transition == str(words[0]+words[1]+words[2])):  
                radex_output["E_up"] = float(words[3]) # Extract the energy of the transition
                radex_output["wav"] = float(words[5]) # Extract the wavelength of the transition
                radex_output["flux"] = float(words[-1]) # Extract the integrated flux of the transition in cgs
    
    return radex_output


# prior probability function 
def ln_prior(x):
    # velocity
    if x[0]<30 or x[0]>60:
        return False
    # density (log space)
    elif x[1]<3 or x[1]>6:
        return False
    # velocity condition for C-type shock (see Jon's UCLCHEM paper)
    elif x[0]>45 and x[1]>=6:
        return False
    else:
        return True


#likelihood function
def ln_likelihood(x, observed_data, database_info, DIREC, RADEX_PATH):

    # Declare a dictionary to hold data
    theoretical_data = {
        'spec': [],
        'N': [],
        'flux': []
    }

    # Pack the parameters in to the y array for emcee
    # 0 is vs, 1 is n (initial density)
    y = [x[0], 10**x[1]]

    # Checks to see whether the randomly selected values of x are within
    # the desired range using ln_prior
    if ln_prior(x):
        
        print("Parameters within prior range")
        #call radex to determine flux of given transitions
        trial_data = get_trial_data(y, observed_data, DIREC, RADEX_PATH)
    
        theoretical_data['spec'] = trial_data['species']
        theoretical_data['flux'] = trial_data['flux']
        theoretical_data['N'] = trial_data['N']
    
        print("Computing chi-squared statistic")
        # Determine chi-squared statistic and write it to file
        chi = chi_squared(theoretical_data['flux'], observed_data['source_flux'],  observed_data['source_flux_error'])

        # Put the data in to a list for easier reference when storing
        data = [observed_data["source"], theoretical_data['flux'], observed_data['source_flux'],
                observed_data['source_flux_error'], chi]

        # Save the best fit data
        db.insert_data(db_params=database_info["bestfit_config_file"],
                       table="{0}-bestfit-conditions".format(observed_data["source"]), data=data)

        return -0.5*chi
    
    else:
        return -np.inf


def chi_squared(observed, expected, error):
    sum = 0
    for indx, val in enumerate(observed):
        sum += ((observed[indx] - expected[indx])**2)/(error[indx]**2)
    return sum


def linewidth_conversion(dv, transition_freq):
    
    # Determine the transition wavelength (in m) from supplied frequency 
    transition_wav = 3e8/(transition_freq)

    # Convert the linewidth to frequency space
    linewidth_f = (dv)/(transition_wav) 

    return linewidth_f


def boltzmann_column_density(nu, int_intensity, A_ul, g_u, Z, E_u, T):
    '''
    Inputs:
        nu: frequency of the transition
        int_intensity: intensity of the line (in K km/s)
        g_u: (2J+1)
        Z: Partition function
        E_u: Upper state energy (in K)
        T: Excitation temperature (in K)
    Outputs:
        N
    '''

    k = 1.38064852e-23 # m2 kg s-2 K-1
    h = 6.62607015e-34 #m2 kg s-1

    # Upper state column density
    N_u = ((8*np.pi*k*(nu**2)*int_intensity*(1e5))/(h*((3e10)**3)*A_ul))
    return (N_u*Z)/(g_u*np.exp(-E_u/T))


def param_constraints(observed_data, sio_data, so_data):

    local_conditions = reset_dict()
    physical_conditions = []

    for indx, spec in enumerate(observed_data['species']):
        # Determine the FWHM (see https://www.cosmos.esa.int/documents/12133/1035800/fts-line-flux-conv.pdf/ or https://www.iram.fr/IRAMFR/ARC/documents/cycle7/ALMA_Cycle7_Technical_Handbook.pdf Page 132)
        resolution = 0.0013 # effective resolution of ALMA band 7 (see https://home.strw.leidenuniv.nl/~alma/memo/allegro_memo3.pdf)

        # Determine the flux conversion factor (again, from https://www.cosmos.esa.int/documents/12133/1035800/fts-line-flux-conv.pdf/)
        Q = 1.22e6 * ((resolution)**-2)*(((observed_data['linewidths'][indx])**-2))

        # Convert flux from Jy km/s to to K km/s
        int_intensity = Q*(observed_data['source_flux_dens_Jy'][indx])*(observed_data['linewidths'][indx])

        # Convert the linewidth in km/s to frequency space
        linewidth_wav = 3e8/(observed_data['transition_freqs'][indx]*(1e9))
        nu = (observed_data['linewidths'][indx]*1e3)/(linewidth_wav) # dv in km/s so convert to m/s

        if spec == "SIO":
            data = sio_data
        elif spec == "SO":
            data = so_data    
            
        A_ul = data['A_ul']
        g_u = data['g_u']
        E_u = data['E_u']
        
        for indx, temp in enumerate(sio_data['T']):
            Z = data['Z'][indx]
            T = data['T'][indx]

            N = boltzmann_column_density(nu, int_intensity, A_ul, g_u, Z, E_u, T)
            
            local_conditions['species'] = spec
            local_conditions['intensity'] = int_intensity
            local_conditions['N'] = N
            local_conditions['T'] = T

            physical_conditions.append(local_conditions)
            
            # Reset the dictionary
            local_conditions = reset_dict()

    return physical_conditions


def reset_dict():
    local_conditions = {
        'species': "",
        'intensity': 0,
        'T': 0,
        'N': 0,
    }
    return local_conditions


def reset_data_dict():
    data = {
        'source': "", # The source name
        'sample_size': 0, # The sample size in beams
        'species': [], # Species of interest
        'transitions': [], # Transitions of interest
        'transition_freqs': [], # Transition frequencies according to Splatalogue in GHz
        'linewidths': [], # Linewidths for the transitions in km/s
        'source_flux_dens_Jy': [], # Flux of species in Jy (original in mJy)
        'source_flux_dens_error_Jy': [], # Error for each flux in Jy (original in mJy)
        'source_flux': [],
        'source_flux_error': []
    }
    return data


def delete_radex_io(species, DIREC):
    for spec in species:
        filelist = glob.glob(os.path.join("{0}/radex-output/{1}/".format(DIREC, spec), "*.out"))
        for file_instance in filelist:
            os.remove(file_instance)
