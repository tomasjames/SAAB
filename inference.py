import glob
import numpy as np
import os
import subprocess as sp
import time

import config
import databasefunctions as db
import workerfunctions


def get_trial_radex_data(params, observed_data, DIREC, RADEX_PATH):

    # Declare dictionary to hold trial data
    trial_data = {
        'species': [],
        'transitions': [],
        'rj_flux': []
    }

    # Unpack the parameters
    try:
        T, n, N_sio, N_so, N_ocs = params[0], params[1], params[2], params[3], params[4]
    except IndexError:
        T, n, N_sio, N_so = params[0], params[1], params[2], params[3]

    for spec_indx, spec in enumerate(observed_data["species"]):
        # Catch any molecular transitions that we're not interested in
        if spec == "SIO" or spec == "SO" or spec == "OCS":
            # Store the species
            trial_data['species'].append(spec)
            trial_data['transitions'].append(observed_data["transitions"][spec_indx])

            # Set the line width and column density
            if spec == "SIO":
                N = N_sio
            elif spec == "SO":
                N = N_so
            elif spec == "OCS":
                N = N_ocs

            dv = abs(observed_data["linewidths"][spec_indx]) # Some linewidths are -ve
            transition = observed_data["transitions"][spec_indx]

            # Write the radex input file
            input_path = '{0}/radex-input/{1}/n{2:E}T{3}N{4:E}.inp'.format(DIREC, spec, n, T, N)
            output_path = '{0}/radex-output/{1}/n{2:E}T{3}N{4:E}.out'.format(DIREC, spec, n, T, N)
            write_radex_input(spec, n, T, n, N, dv, input_path, output_path, RADEX_PATH, f_min=290, f_max=360)

            #Run radex
            shell_output = sp.run(
                'radex < {0}'.format(input_path), 
                shell=True, 
                capture_output=True, 
                check=False
            )  # &> pipes stdout and stderr

            # This block ensures that the output file exists before attempting to read it
            while not os.path.exists(output_path):
                time.sleep(1)
                print("Waiting for {0}".format(output_path))
            if os.path.isfile(output_path):
                # Read the radex output
                radex_output = read_radex_output(spec, transition, output_path)
            else:
                raise ValueError("%s isn't a file!" % output_path)

            # Determine the flux density
            trial_data['rj_flux'].append(radex_output["rj_flux"])
        else:
            continue
    return trial_data



def get_trial_uclchem_data(params, observed_data, DIREC, RADEX_PATH):

    # Declare dictionary to hold trial data
    trial_data = {
        'species': [],
        'transitions': [],
        'resolved_T': 0,
        'resolved_n': 0,
        'N': [],
        'rj_flux': []
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

    # Save those quantities to the dict lists
    trial_data["resolved_T"] = T
    trial_data["resolved_n"] = n

    for spec_indx, spec in enumerate(observed_data["species"]):    
        # Store the species
        trial_data['species'].append(spec)

        # Get the abundances
        abund = shock_model["abundances"][spec_indx][1:]
        
        # Set the column density
        N = workerfunctions.resolved_quantity(
            shock_model["dens"][1:],
            [a*b for a, b in zip(shock_model["H_coldens"][1:], abund)],
            shock_model["times"][1:]
        )
        
        trial_data["N"].append(N)

        # Get the linewidth and relevant transition
        dv = observed_data["linewidths"][spec_indx]
        transition = observed_data["transitions"][spec_indx]

        # Store the transitions
        trial_data['transitions'].append(transition)

        print("Writing the radex inputs")
        input_path = '{0}/radex-input/{1}/vs{2}n{3:.6E}.in\n'.format(DIREC, spec, vs, initial_dens)
        output_path = '{0}/radex-output/{1}/vs{2}n{3:.6E}.out\n'.format(DIREC, spec, vs, initial_dens)
        # Write the radex input file
        write_radex_input(spec, n, T, n, N, dv, input_path, output_path, RADEX_PATH, f_min=290, f_max=360)

        print("Running radex")
        # Run radex
        sp.run('radex < {0}/radex-input/{1}/vs{2}n{3:.6E}.inp &> /dev/null'.format(
            DIREC, spec, vs, initial_dens), shell = True)  # &> pipes stdout and stderr

        # This block ensures that the output file exists before attempting to read it
        while not os.path.exists(output_path):
            time.sleep(1)
            print("Waiting for {0}".format(output_path))
        if os.path.isfile(output_path):
            # Read the radex output
            radex_output = read_radex_output(spec, transition, output_path)
        else:
            raise ValueError("%s isn't a file!" % output_path)

        # Catch any radex saturation problems
        if radex_output["flux"] < 0 or radex_output["flux"] > 1e2:
            radex_output["flux"] = np.inf

        # Append the radex output data to the trial_data dictionary
        trial_data['flux'].append(radex_output["flux"])
    
    return trial_data


# Function to write the input to run Radex
def write_radex_input(spec, ns, tkin, nh2, N, dv, input_path, output_path, RADEX_PATH, f_min, f_max):
    tbg=2.73
    # Open the text file that constitutes Radex's input file
    infile = open(input_path, 'w')
    infile.write('{0}/data/{1}.dat\n'.format(RADEX_PATH,spec.lower()))  # Molecular data file
    infile.write("{0}\n".format(output_path))  # Output file
    infile.write('{0} {1}\n'.format(f_min, f_max)) # Frequency range (0 0 is unlimited)
    infile.write('{0}\n'.format(tkin)) # Kinetic temperature (i.e. temp of gas)
    infile.write('1\n') # Number of collisional partners
    infile.write('H2\n') # Type of collisional partner
    infile.write('{0}\n'.format(nh2)) # Density of collision partner (i.e. dens of gas)
    infile.write('{0}\n'.format(tbg)) # Background temperature
    infile.write('{0}\n'.format(N)) # Column density of emitting species
    infile.write('{0}\n'.format(dv/1e3)) # Line width (converts back to km/s)
    infile.write('0 \n') # Indicates Radex should exit

    infile.close()


def read_radex_output(spec, transition, output_path):

    radex_output = {
        "temp": 0, 
        "dens": 0, 
        "E_up": 0, 
        "wav": 0, 
        "rj_flux": 0
    }

    with open(output_path, 'r') as outfile:
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
                try: 
                    radex_output["rj_flux"] = float(words[-5]) # Extract the Rayleigh-Jeans flux of the transition in K
                except ValueError:
                    print("Potential Radex saturation")
                    radex_output["rj_flux"] = np.inf
    
    return radex_output


# prior probability function
def ln_radex_prior(x):
    #temp
    if x[0] < 100 or x[0] > 2500:
        return False
    #dens (log space)
    elif x[1] < 2 or x[1] > 8:
        return False
    #N_sio (log space)
    elif x[2] < 8 or x[2] > 16:
        return False
    #N_so (log space)
    elif x[3] < 8 or x[3] > 16:
        return False
    if len(x) == 5:
        #N_ocs (log space)
        if x[4] < 8 or x[4] > 16:
            return False
    else:
        return True


# prior probability function 
def ln_uclchem_prior(x):
    # velocity
    if x[0]<5 or x[0]>20:
        return False
    # density (log space)
    elif x[1]<2 or x[1]>5:
        return False
    else:
        return True


def ln_likelihood_radex(x, observed_data, bestfit_config_file, DIREC, RADEX_PATH):

    # Pack the parameters in to the y array for emcee
    # 0 is T, 1 is n and 2 is N_sio and 3 in N_so and 4 is N_ocs
    try:
        y = [x[0], 10**x[1], 10**x[2], 10**x[3], 10**x[4]]
    except IndexError:
        y = [x[0], 10**x[1], 10**x[2], 10**x[3]]

    # Checks to see whether the randomly selected values of x are within
    # the desired range using ln_prior
    if ln_radex_prior(x):
        print("Parameters within prior range")
        #call radex to determine flux of given transitions
        trial_data = get_trial_radex_data(y, observed_data, DIREC, RADEX_PATH)

        print("Computing chi-squared statistic")
        # Determine chi-squared statistic and write it to file
        chi = chi_squared(
            trial_data['rj_flux'],
            observed_data['source_rj_flux'],
            observed_data['source_rj_flux_error']
        )

        if trial_data['rj_flux'] == np.inf:
            print("Radex has potentially saturated")
            trial_data['rj_flux'] = "Infinity"

        try:
            column_densities = [10**x[2], 10**x[3], 10**x[4]]
        except IndexError:
            column_densities = [10**x[2], 10**x[3]]

        # Put the data in to a dictionary for easier reference when storing
        data = {
            "species": trial_data['species'], 
            "transitions": trial_data['transitions'], 
            "temp": x[0], 
            "dens": 10**x[1], 
            "column_density": column_densities,
            "rj_flux": trial_data['rj_flux'], 
            "source_rj_flux": observed_data['source_rj_flux'], 
            "source_rj_flux_error": observed_data['source_rj_flux_error'],
            "chi": chi
        }

        print("Inserting the chain data (and other quantities) in to the database")
        # Save the best fit data for each species
        db.insert_data(
            db_pool=db.dbpool(bestfit_config_file),
            table="{0}_bestfit_conditions".format(observed_data["source"]), 
            data=data
        )

        return -0.5*chi

    else:
        return -np.inf


#likelihood function
def ln_likelihood(x, observed_data, bestfit_db_connection, DIREC, RADEX_PATH):

    # Pack the parameters in to the y array for emcee
    # 0 is vs, 1 is n (initial density)
    y = [x[0], 10**x[1]]

    # Checks to see whether the randomly selected values of x are within
    # the desired range using ln_prior
    if ln_prior(x):
        
        print("Parameters within prior range")
        #call radex to determine flux of given transitions
        trial_data = get_trial_uclchem_data(y, observed_data, DIREC, RADEX_PATH)
    
        print("Computing chi-squared statistic")
        # Determine chi-squared statistic and write it to file
        chi = chi_squared(trial_data['flux'], 
            observed_data['source_flux'],  
            observed_data['source_flux_error']
        )

        if trial_data['flux'] == np.inf:
            print("Radex has potentially saturated")
            trial_data['flux'] = "Infinity"

        # Put the data in to a list for easier reference when storing
        data = [trial_data['species'], trial_data['transitions'], x[0], 10**x[1], 
                    trial_data['resolved_T'], trial_data['resolved_n'], trial_data['N'], 
                    trial_data['flux'], observed_data['source_flux'], observed_data['source_flux_error'], chi]

        print("Inserting the chain data (and other quantities) in to the database")
        # Save the best fit data for each species
        db.insert_data(db_pool=bestfit_db_connection,
                       table="{0}_bestfit_conditions".format(observed_data["source"]), data=data)

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
        int_intensity: intensity of the line (K km/s)
        g_u: (2J+1)
        Z: Partition function
        E_u: Upper state energy (in K)
        T: Gas temperature (in K)
    Outputs:
        N
    '''

    k = 1.38064852e-23 # kg m2 s-2 K-1
    h = 6.62607015e-34 # kg m2 s-1

    # Convert the intensity to K m/s for clarify
    int_intensity = int_intensity*1e3

    # Upper state column density
    N_u = ((8*np.pi*k*(nu**2)*int_intensity)/((h*(3e8)**3)*A_ul))*(1e-2)**2 # (1e-2)**2 converts to cm^-2
    return (N_u*Z)/(g_u*np.exp(-E_u/T))


def param_constraints(observed_data, sio_data, so_data):

    local_conditions = reset_dict()
    physical_conditions = []

    for indx, spec in enumerate(observed_data['species']):

        rj = rj_flux(
            observed_data["source_flux_dens_Jy"][indx], 
            observed_data["transition_freqs"][indx], 
            observed_data["linewidths"][indx]
        )

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

            N = boltzmann_column_density(
                observed_data['transition_freqs'][indx], 
                rj["int_flux_K"], 
                A_ul, 
                g_u, 
                Z, 
                E_u,
                T
            )
            
            local_conditions['species'] = spec
            local_conditions['flux_K'] = rj["rj_flux"]
            local_conditions['int_flux_K'] = rj["int_flux_K"]
            local_conditions['N'] = N
            local_conditions['T'] = T

            physical_conditions.append(local_conditions)
            
            # Reset the dictionary
            local_conditions = reset_dict()

    return physical_conditions


def rj_flux(source_flux_dens_Jy, transition_freq, linewidth):

    # Constants
    k = 1.38064852e-23  # kg m2 s-2 K-1
    h = 6.62607015e-34  # kg m2 s-1

    # Determine the beam solid angle
    d = 2.42e22  #  Distance to Sgr A* in cm
    
    # angular size of x and y dimensions of beam in radians
    theta_x, theta_y = 0.37*4.84814e-6, 0.31*4.84814e-6 
    
    # source area and solid angle
    source_area = np.pi*(theta_x*d)*(theta_y*d)  # small angle approx
    solid_angle = source_area/(d**2)

    # Determine the Rayleigh-Jeans flux in K
    radiant_flux = (source_flux_dens_Jy*1e-26)/solid_angle
    flux_K = ((3e8**2)*radiant_flux) / (2*(transition_freq**2)*k)
    
    # Integrated flux in K km/s
    int_flux_K = flux_K*linewidth

    return {
        "rj_flux": flux_K,
        "int_flux_K": int_flux_K
    }


def reset_dict():
    local_conditions = {
        'species': "",
        'flux_K': 0,
        'int_flux_K': 0,
        'T': 0,
        'N': 0,
    }
    return local_conditions



def delete_radex_io(species, DIREC):
    for spec in species:
        input_filelist = glob.glob(os.path.join("{0}/radex-input/{1}/".format(DIREC, spec), "*.inp"))
        output_filelist = glob.glob(os.path.join("{0}/radex-output/{1}/".format(DIREC, spec), "*.out"))
        for inp, outp in zip(input_filelist, output_filelist):
            os.remove(inp)
            os.remove(outp)


def delete_uclchem_io(DIREC):
    for path in ["data", "start"]:
        filelist = glob.glob(os.path.join(
            "{0}/UCLCHEM/output/{1}/".format(DIREC, path), "*.dat"))
        for file_instance in filelist:
            os.remove(file_instance)

