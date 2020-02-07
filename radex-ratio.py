import glob
from multiprocessing import Pool
import os
import subprocess as sp

from decimal import Decimal
import numpy as np

from chainconsumer import ChainConsumer
import emcee as mc
from random import random

import matplotlib.pyplot as plt

# Define constants
jy = 1e-23
RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"
DIREC = "/Users/tjames/Documents/University/SAAB"


def get_trial_data(params, species):
    transitions, fluxes, specs = [], [], []
    T, n, N_sio, N_so = params[0], params[1], params[2], params[3]
    for spec in species:
        # Set the line width
        if spec == "SO":
            dv = 15.0  # Line width (km/s)
            N = N_sio
        elif spec == "SIO":
            dv = 13.9
            N = N_so
        else:
            dv = 10.0

        # Write the radex input file
        write_radex_input(spec, n, T, n, N, dv, f_min=300, f_max=360)

        #Run radex
        # print('Running RADEX for {0} at n={1:1.0E} and T={2} at N={3:2.1E}'.format(spec,n,T,N))
        sp.run('radex < {0}/radex-input/{1}/n{2:1.0E}T{3}N{4:2.1E}.inp &> /dev/null'.format(
            DIREC, spec, int(n), int(T), N), shell=True) # &> pipes stdout and stderr

        # Delete the input file (no longer required)
        os.remove("{0}/radex-input/{1}/n{2:1.0E}T{3}N{4:2.1E}.inp".format(
            DIREC, spec, int(n), int(T), N))

        # Read the radex input
        outfile = open('{0}/radex-output/{1}/n{2:1.0E}T{3}N{4:2.1E}.out'.format(
                                DIREC, spec, int(n), int(T), N), 'r')
        temp, dens, transition, E_up, nu, flux = read_radex_output(outfile)

        # Delete the output file (no longer required)
        os.remove("{0}/radex-output/{1}/n{2:1.0E}T{3}N{4:2.1E}.out".format(
            DIREC, spec, int(n), int(T), N))

        specs.append(species)        
        transitions.append(transition)
        fluxes.append(flux)

    return specs, transitions, fluxes


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


def read_radex_output(outfile):
    transition, fluxes, nu, E_up = [], [], [], []
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
            nu.append(float(words[4])) # Extract the frequency of the transition
            fluxes.append(float(words[-1])) # Extract the emitted photon flux of the transition
    
    return temp, dens, transition, E_up, nu, fluxes


# prior probability function 
def ln_prior(x):
    #temp
    if x[0]<10 or x[0]>500:
        return False
    #dens (log space)
    elif x[1]<3 or x[1]>6:
        return False
    #N_sio (log space)
    elif x[2]<11 or x[2]>16:
        return False
    #N_so (log space)
    elif x[3]<11 or x[3]>16:
        return False
    else:
        return True


#likelihood function
def ln_likelihood(x, observed_data, observed_data_error, species):
    # Pack the parameters in to the y array
    # 0 is T, 1 is n and 2 is N_sio and 3 in N_so
    y = [x[0], 10**x[1], 10**x[2], 10**x[3]]

    # Checks to see whether the randomly selected values of x are within
    # the desired range using ln_prior
    if ln_prior(x):
        #call radex and load arrays of fluxes
        specs, transitions, fluxes = get_trial_data(y, species)
        
        # Determine the intensity ratio by finding the correct
        # species and transitions and dividing the two
        for i in range(len(fluxes)):
            for j in range(len(fluxes[i])):
                if (transitions[i][j] == "7--6"):
                    SO_flux = fluxes[i][j]
                if (transitions[i][j] == "8_7--7_6"):
                    SIO_flux = fluxes[i][j]

        theoretical_ratio = SIO_flux/SO_flux
        ratio_file.write("%f %f %f %f %f\n" % (theoretical_ratio, x[0], x[1], x[2], x[3]))

        # Determine Chi-squared statistic
        chi = chi_squared(observed_data, theoretical_ratio, observed_data_error)
        
        return -0.5*chi
    
    else:
        return -np.inf


def chi_squared(observed, expected, error):
    # print(expected)
    return ((observed - expected)**2)/((error)**2)



if __name__ == '__main__':
    # Define Farhad's data
    source_spec = ["SIO","SO"] # Species of interest
    source_ratio = 0.95
    source_ratio_error = 0.27

    species = ['SIO','SO']
    sio_flux, so_flux = [], []
    sio_v, so_v = [], []
    ratio, best_fit_ratio, best_fit_index = [], [], []
    params = []

    continueFlag = False
    nWalkers = 8
    nDim = 4 # Number of dimensions within the parameters
    nSteps = int(1e5)
    
    # TODO output the ratio to a file so as to check the ratio manually        
    ratio_file = open("{0}/radex-output/ratios.csv".format(DIREC),"w")

    with Pool() as pool:
        sampler = mc.EnsembleSampler(nWalkers, nDim, ln_likelihood, args=[source_ratio, source_ratio_error, species], pool=pool)
        pos = []
        f = []

        if not os.path.exists('{0}/chains/'.format(DIREC)):
            os.makedirs('{0}/chains/'.format(DIREC))
        if not continueFlag:
            for i in range(nWalkers):
                dens=((random()*5.0)+3.0)
                N_sio=((random()*6.0)+11.0)
                N_so=((random()*6.0)+11.0)
                T=10+(random()*45.0)
                pos.append([T,dens,N_sio,N_so])
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
            pos,prob,state = sampler.run_mcmc(pos, nBreak, progress=True) #start from where we left off previously 
            chain = np.array(sampler.chain[:,:,:]) #get chain for writing
            chain = chain.astype(float)
            #chain is organized as chain[walker,step,parameter]
            for i in range(0,nWalkers):
                for j in range(0,nBreak):
                    outString = ""
                    for k in range(0,nDim):
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
        file_out = "{0}/radex-plots/corner.pdf".format(DIREC)
        c = ChainConsumer() 
        c.add_chain(chain, parameters=[param1,param2,param3,param4], walkers=nWalkers)
        c.configure(statistics="max", color_params="posterior")
        fig = c.plotter.plot(filename=file_out, display=False)
        summary = c.analysis.get_summary()
