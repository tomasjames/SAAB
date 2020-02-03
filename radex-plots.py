from plotfunctions import *
import glob
import os
import numpy as np

import matplotlib.pyplot as plt

# Define constants
jy = 1e-23
RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"
DIREC = "/Users/tjames/Documents/University/sgrA/UCLCHEM/scripts"


def read_radex(outfile):
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
            fluxes.append(float(words[-1]))
    return temp, dens, transition, E_up, nu, fluxes

# Define Farhad's data
source_spec = ["SIO"] # Species of interest
source_nu = [303.89] # frequency in GHz
source_T_l = [16.3] # Tl in mJy
source_T_l_err = [3.1] # Error in Tl

species = ['SIO']
vs = [30, 60]
ns = [1e3, 1e6]
times = np.linspace(1e3,1e4,10)

for spec in species:
    for time in times:
        for v in vs:
            for n in ns:
                try:
                    outfile = open('{0}/radex-output/{1}/v{2}n{3}/t{4}.out'.format(DIREC,spec,int(v),int(n),int(time)))
                except FileNotFoundError:
                    continue
                
                # Read Radex data
                temp, dens, transitions, E_up, nu, flux = read_radex(outfile)

                for indx, transition in enumerate(transitions):
                    print(transition)
                    if transition != "7--6":
                        continue
                    else:
                        print(flux)
                        print('--------')

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

                    
