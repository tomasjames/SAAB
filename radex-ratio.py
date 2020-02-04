import glob
import os

from decimal import Decimal
import math
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
    T, n, N = params[0], params[1], params[2]
    for spec in species:
        # Set the line width
        if spec == "SO":
            dv = 15.0  # Line width (km/s)
        elif spec == "SIO":
            dv = 13.9
        else:
            dv = 10.0

        # Write the radex input file
        write_radex_input(spec, n, T, n, N, dv, f_min=300, f_max=360)

        #Run radex
        print('Running RADEX for {0} at n={1:1.0E} and T={2} at N={3:2.1E}'.format(spec,n,T,N))
        os.system('radex < {0}/radex-input/{1}/n{2:1.0E}T{3}N{4:2.1E}.inp\n > /dev/null'.format(
            DIREC, spec, int(n), int(T), N))

        # Read the radex input
        outfile = open('{0}/radex-output/{1}/n{2:1.0E}T{3}N{4:2.1E}.out'.format(
                                DIREC, spec, int(n), int(T), N), 'r')
        temp, dens, transition, E_up, nu, flux = read_radex_output(outfile)

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
    if x[0]>=10 or x[0]<=500:
        return True
    #dens
    elif x[1]>=3 or x[1]<=7:
        return True
    #N
    elif x[2]>=11 or x[2]<=16:
        return True
    else:
        return False


#likelihood function
def ln_likelihood(x, observed_data, observed_data_error, species):
    # Pack the parameters in to the y array
    # 0 is T, 1 is n and 2 is N 
    y = x[0], 10**x[1], 10**x[2]

    #call radex and load arrays of fluxes
    specs, transitions, fluxes = get_trial_data(y, species)
    
    # Determine the intensity ratio by finding the correct
    # species and transitions
    print(transitions)
    for i in range(len(fluxes)):
        for j in range(len(fluxes[i])):
            print(i)
            print(j)
            print(transitions[i][j])
            if (transitions[i][j] == "7--6"):
                SO_flux = fluxes[i][j]
            if (transitions[i][j] == "8_7--7_6"):
                SIO_flux = fluxes[i][j]
    
    theoretical_ratio = SIO_flux/SO_flux

    # Determine Chi-squared statistic
    chi = chi_squared(observed_data, theoretical_ratio, observed_data_error)
    
    return -0.5*chi


def chi_squared(observed, expected, error):
    print(expected)
    return ((observed - expected)**2)/((error)**2)


# Define Farhad's data
source_spec = ["SIO","SO"] # Species of interest
source_ratio = 0.95
source_ratio_error = 0.27

species = ['SIO','SO']
sio_flux, so_flux = [], []
sio_v, so_v = [], []
ratio, best_fit_ratio, best_fit_index = [], [], []
params = []

temp = [10, 500]
ns = np.logspace(3, 6, 2)
coldens = np.logspace(11, 15, 2)

nWalkers = 6
nDim = 3 # Number of dimensions within the parameters
nSteps = int(3e4)
sampler = mc.EnsembleSampler(nWalkers, nDim, ln_likelihood, args=[source_ratio, source_ratio_error, species])
pos = []

for i in range(nWalkers):
    dens=((random()*5.0)+3.0)
    N=((random()*6.0)+11.0)
    T=10+(random()*45.0)
    pos.append([T,dens,N])

nBreak=int(nSteps/10)
for counter in range(0,10):
    sampler.reset() #lose the written chain
    pos, prob, state = sampler.run_mcmc(pos, nBreak) #start from where we left off previously 
    chain = np.array(sampler.chain[:,:,:]) #get chain for writing
    chain = chain.astype(float)
    #chain is organized as chain[walker,step,parameter]
    for i in range(0,nWalkers):
        for j in range(0,nBreak):
            outString=""
            for k in range(0,nDim):
                outString += "{0:.5f}\t".format(chain[i][j][k])
            f[i].write(outString+"\n")
    print("{0:.0f}%".format((counter+1)*10))

sampler.reset()

'''
for n in ns:
    for T in temp:
        for N in coldens:
            for specIndx, spec in enumerate(species):
                try:
                    outfile = open('{0}/radex-output/{1}/n{2:1.0E}T{3}N{4:2.1E}.out'.format(
                                DIREC, spec, int(n), int(T), N), 'r')
                except FileNotFoundError:
                    break

                # Read Radex data
                T_kin, dens, transitions, E_up, nu, flux = read_radex_output(outfile)

                for indx, transition in enumerate(transitions):
                    if (spec == "SIO" and transition == "7--6"):
                        sio_flux = flux[indx]
                    elif (spec == "SO" and transition == "8_7--7_6"):
                        so_flux = flux[indx]

            # Determine line ratio
            if sio_flux == 0.0 or so_flux == 0.0:
                break
            else:
                ratio.append(sio_flux/so_flux)
                # Store physical parameters for later
                params.append([n, T, N])

stats = []

# Determine best fit and likelihood function
iter = -1
for indn, n in enumerate(ns):
    for indT, T in enumerate(temp):
        for indN, N in enumerate(coldens):
            iter += 1
            chi = chi_squared(source_ratio, ratio[iter], source_ratio_error)
            # Ensure that if chi is NaN, it's set to a large enough number not to be 
            # mistaken
            if math.isnan(chi):
                stats.append(1e10)
            else:
                stats.append(chi)




bfi = stats.index(np.min(stats))
best_fit_index.append(bfi)
best_fit_ratio.append(ratio[int(p)][int(bfi)])
stats = []

# Plot the best fitting parameters
for count,index in enumerate(best_fit_index):
    n = str(int(params[index][0]))
    n = '%.2E' % Decimal(n)
    n_power = len(n)-len(n.rstrip('0'))
    v = int(params[index][1])
    if times[count] == 4000.0:
        continue
    else:
        plt.plot(times[count], best_fit_ratio[count], linestyle="None", marker='x', label="v={0} km/s  n={1} /cm3".format(str(v), str(n)))

# Plot UCLCHEM
# sim_times, dens, temp, av, specAbundances = read_uclchem("../output/cshock/data/v{}", ["SO", "SIO"])

plt.plot(times, np.linspace(source_ratio,source_ratio,len(times)), linestyle='--', label="Observed ratio")
# plt.axvspan(xmin=times[0], xmax=times[-1], ymin=0.56, ymax=0.64, color='r', alpha=0.5)
plt.xlabel("t (years)")
plt.ylabel("I$_{SiO}$/I$_{SO}$")
plt.title("J1")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("ratios.png", dpi=500)
plt.close()

# Format the transition string
transition_fmt = '$' + \
    transition[:transition.find(
        '-')] + '$--$' + transition[transition.find('-')+2:] + '$'

ax = plt.subplot()
ax.set_yscale('log')
plt.plot(nu[indx], flux[indx], '+', label='{0}({1}) RT'.format(str(spec), transition_fmt))
plt.errorbar(source_nu, source_T_l, yerr=source_T_l_err, marker='x', label='{0} N7b'.format(spec))
plt.xlabel('$f$ (GHz)')
plt.ylabel('$F$ (mJy)')
plt.legend(loc='best')
plt.savefig(
    '{0}/radex-plots/{1}/v{2}n{3}/{4}-t{5}.pdf'.format(DIREC, spec, int(v), int(n), str(transition), int(time)))
plt.close()
'''