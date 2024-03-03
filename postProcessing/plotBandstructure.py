import os
import numpy as np
import periodictable as pdt

import pyprocar
from pyprocar.scripts import *
    
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re

hartree_to_ev = 27.211386024367243
scale_factor = 0.5291772105638411 # scale factor (bohr to angstrom)

def plotBandStr(filesPath, bandsDatFile,kptsFile,coordinatesFile,latticeVecFile, elimit, isPeriodic):
    """
    Generates a band structure plot for a given system.

    Parameters:
    filesPath (str): The path to the directory containing the necessary files.
    bandsDatFile (str): The name of the file containing band data.
    kptsFile (str): The name of the file containing k-points data.
    coordinatesFile (str): The name of the file containing atomic coordinates.
    latticeVecFile (str): The name of the file containing lattice vectors.
    elimit (list): The energy limits for the plot, specified as a list of two numbers.
    isPeriodic (bool): A flag indicating whether the system is periodic.

    This function reads data from the provided files, generates a POSCAR file, a PROCAR file, and an OUTCAR file, 
    and finally generates a band structure plot. The plot is saved as 'bandsplot.png' in the 'outfiles' directory 
    within the directory specified by 'filesPath'.

    Note: This function does not return a value. It performs its operations through side effects (creating files, 
    generating a plot).
    """

    if isPeriodic:
        ionPosVecType = "Direct"
    else:
        ionPosVecType = "Cartesian"
        
    outdir = filesPath +"outfiles/"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with open(outdir+'POSCAR', 'w') as f:
        f.write("This is a commented line\n") # Here you can write the name of the system (it's a commented line in POSCAR file)
        with open(filesPath+ latticeVecFile) as f1:
            f.write("{}\n".format(scale_factor)) 
            lines = f1.readlines()
            for line in lines:
                f.write(line)
                
        with open(filesPath+coordinatesFile) as f1:
            lines = f1.readlines()
            lines = list(map(lambda x: x.strip(), lines))
            while '' in lines: # To remove the blank lines
                lines.remove('')
            
            atomNumCount ={}

            
            for line in lines:
                line = line.strip()
                atomNum = int(line.strip().split()[0])
                
                if atomNum in atomNumCount.keys():
                    atomNumCount[atomNum] += 1
                else:
                    atomNumCount[atomNum] = 1
            
            for key in atomNumCount.keys():
                f.write("{} ".format(pdt.elements[key]))
            
            f.write("\n")
            
            for value in atomNumCount.values():
                f.write("{} ".format(value))
            
            f.write("\n{}\n".format(ionPosVecType))
            
            for line in lines:
                atomNum = int(line.strip().split()[0])
                newLine = ' '.join(line.strip().split()[2:5])
                newLine += ' {}\n'.format(pdt.elements[atomNum])
                f.write(newLine)
                
    num_ions = 0

    with open(filesPath+coordinatesFile) as f:
        for line in f:
            line = line.strip() #this is done to check if the line is empty or not
            if len(line)!=0:
                num_ions +=1
                
    kptW = []

    with open(filesPath + kptsFile) as f:
        for line in f:
            line = line.replace(',',' ')
            kptW.append(line.strip().split()[:4])
            

    # create PROCAR file

    with open(outdir + "PROCAR", "w") as f:
        f.write("This is a commented line\n") 
        with open(filesPath+ bandsDatFile) as f1:
            
            line = f1.readline()
            numKpts, numBandsPerKpt = list(map(int,line.strip().split()[:2]))
            eFermi  = float(line.strip().split()[-1])  
            f.write("# of k-points:  {}         # of bands:   {}         # of ions:    {}\n\n".format( numKpts,numBandsPerKpt,num_ions))
            
            for line in f1:
                
                l = list(map(float,line.strip().split()))
                k, b, e, occ = int(l[0]), int(l[1]), l[2], l[3]
                
                if (b) % (numBandsPerKpt) == 0:
                    f.write (" k-point     {} :    {} {} {}     weight = {}\n\n".format(k+1, kptW[k][0], kptW[k][1], kptW[k][2], kptW[k][3]))
                
                f.write ("band     {} # energy   {} # occ.  {}\n\n".format(b+1, e * hartree_to_ev, occ ))
                f.write ("ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot\n")
                
                for i in range(num_ions):
                    f.write(str(i+1)+"    0 "*10 + "\n")  # for now all are taken as 0, later to be changed to actual values
                f.write("tot {} \n\n".format("    0 "*10))
                
    # create OUTCAR file

    with open(outdir + "OUTCAR","w") as f:
        f.write(" E-fermi :   {}".format(eFermi*hartree_to_ev)) # Only the Fermi energy part from OUTCAR is needed for bandstructure
        
        
    # final plotting

    splKticks =[]

    if ("kticks" not in globals() and "knames" not in globals()):
        kticks = []
        knames = []
        with open(filesPath + kptsFile) as f:
            for lineNum,line in enumerate(f):
                if '#' in line:
                    kticks.append(lineNum)
                    knames.append(re.split('#', line)[-1])

                if '|' in re.split('#', line)[-1]:
                        splKticks.append(lineNum)
                        
    gph = pyprocar.bandsplot(
                    code='vasp',
                    mode='plain',
                    show = False,
                    elimit = elimit,
                    dirname = outdir)
                    
    if len(splKticks) !=0: 
        for i in range(numBandsPerKpt):
            xdat = gph[1].get_lines()[i].get_xdata()
            for pt in splKticks:
                xdat[pt+1] = xdat[pt]
                try:
                    for j in range(pt+2, len(xdat)):
                        xdat[j] = xdat[j]-1 
                except IndexError:
                    pass
            gph[1].get_lines()[i].set_xdata(xdat)
            
        for pt in splKticks:     
            for k in range(len(kticks)):
                if kticks[k] > xdat[pt +1]:
                    kticks[k] = kticks[k] - 1
            


    if kticks and knames:
        gph[1].set_xticks(kticks, knames)
        for x in kticks:
            gph[1].axvline(x, color='k', linewidth = 0.01)  # Add a vertical line at xticks values

    gph[1].set_xlim(None, kticks[-1])   
            
    gph[1].yaxis.set_major_locator(MultipleLocator(1.0))
    gph[1].grid(True)

    gph[0].savefig(outdir+'bandsplot.png', dpi = 500)
