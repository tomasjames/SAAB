import glob
from multiprocessing import Pool
import os
import subprocess as sp

from astropy import units as u
from decimal import Decimal
import numpy as np

from chainconsumer import ChainConsumer
import emcee as mc
import random

import matplotlib.pyplot as plt

# Define constants
DIREC = os.getcwd()
RADEX_PATH = "{0}/../Radex".format(DIREC)


def get_trial_data(params, species, transitions, transition_freqs):
    # Declare dictionary to hold trial data
    trial_data = {
        'species': [],
        'flux_dens': []
    }

    # Unpack the parameters
    T, n, N_sio, N_so = params[0], params[1], params[2], params[3]
    
    for spec_indx, spec in enumerate(species):
        
        # Store the species
        trial_data['species'].append(spec)

        # Set the line width and column density
        if spec == "SIO":
            dv = 11.3  # Line width (km/s)
            N = N_sio
        elif spec == "SO":
            dv = 9.23
            N = N_so
        else:
            dv = 17.5

        # Write the radex input file
        write_radex_input(spec, n, T, n, N, dv, f_min=300, f_max=360)

        #Run radex
        sp.run('radex < {0}/radex-input/{1}/n{2:1.0E}T{3}N{4:2.1E}.inp &> /dev/null'.format(
            DIREC, spec, int(n), int(T), N), shell=True) # &> pipes stdout and stderr

        # Read the radex input
        temp, dens, transitions, E_up, wav, flux = read_radex_output(spec, n, T, N)

        # Determine the transition wavelength (in m) from supplied frequency 
        # (this is more accurate than the Radex determined transition frequency)
        transition_wav = 3e8/(transition_freqs[spec_indx]*1e9)
        print("transition_wav=", transition_wav)

        # Convert the linewidth to frequency space
        linewidth_f = (dv*1e3)/(transition_wav) # dv in km/s so convert to m/s
        print("linewidth_f=", linewidth_f)
        
        # Determine the flux density
        print("flux[spec_indx]=",flux[spec_indx])
        print("linewidth_f=", linewidth_f)
        print("flux[spec_indx]/linewidth_f=",flux[spec_indx]/linewidth_f)
        trial_data['flux_dens'].append(flux[spec_indx]/linewidth_f) # Converts the fluxes to flux density in ergs/cm2/s/Hz

    return trial_data


# Function to write the input to run Radex
def write_radex_input(spec, ns, tkin, nh2, N, dv, f_min, f_max, vs=None, t=None):
    tbg=2.73
    # Open the text file that constitutes Radex's input file
    if vs==None and t==None:
        infile = open('{0}/radex-input/{1}/n{2:1.0E}T{3}N{4:2.1E}.inp'.format(
            DIREC, spec, int(ns), int(tkin), N), 'w')
    else:
        infile = open('{0}/radex-input/{1}/v{2}n{3:1.0E}/t{4}.inp'.format(
            DIREC, spec, int(vs), int(ns), int(t)), 'w')   
    infile.write('{0}/data/{1}.dat\n'.format(RADEX_PATH,spec.lower()))  # Molecular data file
    if vs==None and t==None:
        infile.write('{0}/radex-output/{1}/n{2:1.0E}T{3}N{4:2.1E}.out\n'.format(
            DIREC, spec, int(ns), int(tkin), N))  # Output file    
    else:
        infile.write('{0}/radex-output/{1}/v{2}n{3:1.0E}/t{4}.out\n'.format(DIREC,spec,int(vs),int(ns),int(t)))  # Output file
    infile.write('{0} {1}\n'.format(f_min,f_max)) # Frequency range (0 0 is unlimited)
    infile.write('{0}\n'.format(tkin)) # Kinetic temperature (i.e. temp of gas)
    infile.write('1\n') # Number of collisional partners
    infile.write('H2\n') # Type of collisional partner
    infile.write('{0}\n'.format(nh2)) # Density of collision partner (i.e. dens of gas)
    infile.write('{0}\n'.format(tbg)) # Background temperature
    infile.write('{0}\n'.format(N)) # Column density of emitting species
    infile.write('{0}\n'.format(dv)) # Line width
    infile.write('0 \n') # Indicates Radex should exit


def read_radex_output(spec, n, T, N):
    transition, fluxes, wav, E_up = [], [], [], []
    outfile = open('{0}/radex-output/{1}/n{2:1.0E}T{3}N{4:2.1E}.out'.format(
                    DIREC, spec, int(n), int(T), N), 'r')
    lines = outfile.readlines()
    for line in lines:
        words = line.split()
        # Extract kinetic temperature
        if (words[1] == "T(kin)"): 
            temp = float(words[-1])
        # Extract density
        if (words[1] == "Density"):
            dens = float(words[-1])
        # Extract the RADEX data (easiest is to recognise when -- is used to 
        # define the line transition)
        if (words[1] == "--"):  
            transition.append(str(words[0])+str(words[1])+str(words[2])) # Extract the transition
            E_up.append(float(words[3])) # Extract the energy of the transition
            wav.append(float(words[4])) # Extract the frequency of the transition
            fluxes.append(float(words[-1])) # Extract the integrated flux of the transition in cgs
    
    # Delete the input file (no longer required)
    os.remove("{0}/radex-input/{1}/n{2:1.0E}T{3}N{4:2.1E}.inp".format(
        DIREC, spec, int(n), int(T), N))

    # Delete the output file (no longer required)
    os.remove("{0}/radex-output/{1}/n{2:1.0E}T{3}N{4:2.1E}.out".format(
        DIREC, spec, int(n), int(T), N))

    return temp, dens, transition, E_up, wav, fluxes


# prior probability function 
def ln_prior(x):
    #temp
    if x[0]<10. or x[0]>1000.:
        return False
    #dens (log space)
    elif x[1]<0. or x[1]>7.:
        return False
    #N_sio (log space)
    elif x[2]<9. or x[2]>16.:
        return False
    #N_so (log space)
    elif x[3]<9. or x[3]>16.:
        return False
    else:
        return True


#likelihood function
def ln_likelihood(x, observed_data, observed_data_error, species, transitions, transition_freqs):

    # Declare a dictionary to hold data
    theoretical_data = {}
    # Define the dictionary elements
    theoretical_data['spec'] = []
    theoretical_data['flux_dens'] = []

    # Pack the parameters in to the y array for emcee
    # 0 is T, 1 is n and 2 is N_sio and 3 in N_so
    y = [x[0], 10**x[1], 10**x[2], 10**x[3]]

    # Checks to see whether the randomly selected values of x are within
    # the desired range using ln_prior
    if ln_prior(x):

        #call radex to determine flux of given transitions
        trial_data = get_trial_data(y, species, transitions, transition_freqs)
    
        theoretical_data['spec'] = trial_data['species']
        theoretical_data['flux_dens'] = [(data/1e-23) for data in trial_data['flux_dens']] # Converts from ergs/cm2/s/Hz to Jy

        # Determine 3-squared statistic and write it to file
        chi = chi_squared(theoretical_data['flux_dens'], observed_data,  observed_data_error)
        prediction_file.write("{0} \t {1} \t {2} \t {3} \t {4} \t {5} \t {6} \n".format(theoretical_data['flux_dens'][0], theoretical_data['flux_dens'][1], chi, x[0], x[1], x[2], x[3]))

        return -0.5*chi
    
    else:
        return -np.inf


def chi_squared(observed, expected, error):
    sum = 0
    for indx, val in enumerate(observed):
        sum += ((observed[indx] - expected[indx])**2)/(error[indx]**2)
    return sum



if __name__ == '__main__':
    
    # Define Farhad's data (temporary - need to compute from saved file)
    observed_data = {
        'species': ["SIO", "SO"], # Species of interest
        'transitions': ["7--6", "8_7--7_6"], # Transitions of interest
        'transition_freqs': [303.92696000, 304.07791400], # Transition frequencies according to Splatalogue
        'source_flux_Jy': [14.3/1e3, 18.5/1e3], # Flux of species in Jy
        'source_flux_error_Jy': [2.9/1e3, 3.6/1e3] # Error for each flux in mJy
    }

    '''
    species = ["SIO", "SO"] # Species of interest
    transitions = ["7--6", "8_7--7_6"] # Transitions of interest
    transition_freqs = [303.92696000, 298.25789470] # Transition frequencies according to Splatalogue
    source_flux_mJy = [14.3, 18.5] # Flux of species in mJy
    source_flux_error_mJy = [2.9, 3.6] # Error for each flux in mJy
    source_flux_Jy = [flux/1e-3 for flux in source_flux_mJy] 
    source_flux_error_Jy = [flux_error/1e-3 for flux_error in source_flux_error_mJy]
    '''

        source_flux_error_K.append((source_flux_error_mJy[specIndx]*u.mJy/source_beam_area).to(
                u.K, equivalencies = equiv).value)
    '''
    
    continueFlag = False
    nWalkers = 8
    nDim = 4 # Number of dimensions within the parameters
    nSteps = int(1e2)
    
    prediction_file = open("{0}/radex-output/predictions.csv".format(DIREC),"w")
    
    pool = Pool(cpu=4)
    
    sampler = mc.EnsembleSampler(
        nWalkers, nDim, ln_likelihood, 
        args=[
            [observed_data['source_flux_Jy']], 
            [observed_data['source_flux_error_Jy']], 
            observed_data['species'], 
            observed_data['transitions'], 
            observed_data['transition_freqs']
        ], 
        pool=pool)
    
    pos = []
    f = []

    if not os.path.exists('{0}/chains/'.format(DIREC)):
        os.makedirs('{0}/chains/'.format(DIREC))
    if not continueFlag:
        for i in range(nWalkers):
            dens = ((random.random()*5.0)+2.0)
            N_sio = ((random.random()*7.0)+11.0)
            N_so = ((random.random()*7.0)+11.0)
            T = 10+(random.random()*990.0)
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
        pos, prob, state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 
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
    # param1 = "T"
    # param2 = "n"
    # param3 = "SiO"
    # param4 = "SO"

    #Chain consumer plots posterior distributions and calculates
    #maximum likelihood statistics as well as Geweke test
    file_out = "{0}/radex-plots/corner.pdf".format(DIREC)
    c = ChainConsumer() 
    c.add_chain(chain, parameters=[param1,param2,param3,param4], walkers=n_walkers)
    c.configure(color_params="posterior", cloud=True, usetex=True, summary=False) 
    fig = c.plotter.plot(filename=file_out, display=False)
    # fig.set_size_inches(6 + fig.get_size_inches())
    # summary = c.analysis.get_summary()
