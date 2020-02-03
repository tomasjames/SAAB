from plotfunctions import *
import glob
import os
import numpy as np

# Define constants
amu = 1.66053904e-27 # kg
k = 1.38064852e-23 # J/K
RADEX_PATH = "/Users/tjames/Documents/Codes/Radex"
DIREC = "/Users/tjames/Documents/University/sgrA/UCLCHEM/scripts"

# Function to write the input to run Radex
def write_input(spec, ns, tkin, nh2, f_min, f_max, N, dv, vs=None, t=None):
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

# Create lists for all files to run Radex on
# as well as the species with which to investigate
temp = [10, 100, 250, 500]
ns = np.logspace(3, 6, 4)
coldens = np.logspace(11, 15, 5)

# Define the times at which to sample shock parameters
times = np.linspace(1e3,1e4,10)

# Define the species and their masses
species = ["SO","SIO","OH2CS","PH2CS"]
masses = [48*amu, 44*amu, 46*amu]

# Define constants for Radex
tbg = 2.73 # Background temperature (K)
f_min = 260 # Minimum frequency (Hz)
f_max = 360  # Maximum frequency (Hz)

for indT, T in enumerate(temp):
    for indn, n in enumerate(ns):
        for indN, N in enumerate(coldens):
            # file = "../output/cshock/data/n{1}T{2}rad10z1.dat".format(int(n),int(T))
            for specIndx, spec in enumerate(species):

                # Account for the differing conventions with species name in UCLCHEM and RADEX
                # i.e. UCLCHEM does not handle ortho/para forms of molecules
                # RADEX does
                if ((spec == "OH2CS") or (spec == "PH2CS")):
                    specName = "H2CS"
                else:
                    specName = spec
                
                # Set the line width
                if spec == "SO":
                    dv = 15.0  # Line width (km/s)
                elif spec == "SIO":
                    dv = 13.9
                else:
                    dv = 10.0
                
                # Read in from static evolution of cloud through phase 2 of UCLCHEM
                # This is to correctly set the initial chemistry and therefore column density
                # of relevant species
                # ctime,cdens,ctemp,cAv,cabundances=read_uclchem("../output/data/n{0}T{1}rad10z1.dat".format(int(n),int(T)),[specName])

                # Write the radex input file
                write_input(spec, n, T, n, f_min, f_max, N, dv)

                print('Running RADEX for {0} at n={1:1.0E} and T={2} at N={3:2.1E}'.format(spec,n,T,N))
                os.system('radex < {0}/radex-input/{1}/n{2:1.0E}T{3}N{4:2.1E}.inp\n > /dev/null'.format(
                    DIREC, spec, int(n), int(T), N))

                '''
                # Read the UCLCHEM data
                sim_times, dens, temp, av, specAbundances = read_uclchem(file, specName)

                # Loop through times
                for time in times:
                    # Select T and n to run Radex for
                    timeIndx = (np.abs(sim_times - time)).argmin()
                    T_radex = float(temp[timeIndx])
                    n_radex = float(dens[timeIndx])
                    av_radex = float(av[timeIndx])

                    # Determine column density 
                    # This is H column density multiplied by fractional abundance
                    coldens = (av_radex*1.6e21)*specAbundances[0][timeIndx]

                    # Write the radex input file
                    write_input(spec, v, n, T_radex, n_radex, time, f_min, f_max, coldens, dv)

                    print('Running RADEX for {0} at v={1}, n={2} at t={3}'.format(spec,v,n,time))
                    os.system('radex < {0}/radex-input/{1}/v{2}n{3}/t{4}.inp > /dev/null'.format(
                        DIREC, spec, int(v), int(n), int(time)))
                '''