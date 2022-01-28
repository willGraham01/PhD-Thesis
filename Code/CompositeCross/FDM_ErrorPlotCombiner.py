#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:57:24 2022

@author: will

Script to place FDM error plots onto the same axis, with appropriate labels.
This file handles outputs saved in the format of FDM_DirichletEval_ConvInvestigation.py.
"""

import argparse
import sys
import glob

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from datetime import datetime

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='FDM convergence rate plots: Reads in multiple datasets of the error in the eigenvalue against log(N), plotting them all on the same axis. Outputs should be in the format saved by FDM_DirichletEval_ConvInvestigation.py')
    parser.add_argument('files', nargs='+', help='<Required> Filenames of .npz archives to extract data from and plot on same axis. Regular expressions supported by glob are allowed, search will not be recursive.')
    parser.add_argument('-fOut', default='./', type=str, help='<Default .> File location to save plot outputs to.')
    parser.add_argument('-logH', action='store_true', help='If passed, the convergence rate plot will be created with the log of the error against the mesh width h rather than the number of mesh points N.')
    parser.add_argument('-p', action='store_true', help='If passed, compute the line of best fit through each set of data, and save to an output file.')

    args = parser.parse_args()
    
    # gather all .npz archives that were provided
    allFiles = []
    for expr in args.files:
        allFiles.extend( glob.glob(expr, recursive=False) )

    # get timestamp for saving plots later
    now = args.fOut + 'FDM_' + datetime.today().strftime('%Y-%m-%d-%H-%M')
    
    # dictionary to store all plot data
    results = {}
    # load all the data that we require
    for file in allFiles:
        load = np.load(file)
        results[int(load['n']), int(load['m'])] = [ load['NVals'], load['sqErrors']]

#%% Create the plot that was requested

    # plot the error
    # do we want log(N) or log(h)?
    logFig, logAx = plt.subplots(1)
    logAx.set_ylabel(r'$\log\left(\vert (n^2+m^2)\pi^2 - \omega_N^2 \vert\right)$')
    
    if args.logH and args.p:
        # plot vs log H and record line of best fit data
        bestFitFile = now + '_LoBF-h.txt'
        with open(bestFitFile, 'a') as f:
            f.write('Estimated ln(Error) = k_0 + k_1*ln(h) \n')
            f.write('n, m, k_0, k_1 \n')
        logAx.set_xlabel(r'Log of mesh width, $\log(h)$')
        # plot all sets of data on same axis, and record polyfit data
        for key in results.keys():
            logAx.plot( np.log(1./(results[key][0]-1)), np.log(results[key][1]), label=r'$n=%d, m=%d$' % (key[0], key[1]) )
            polyFit = np.polyfit(np.log(1./(results[key][0]-1)), np.log(results[key][1]), 1)
            with open(bestFitFile, 'a') as f:
                f.write('%d, %d, %.8f, %.8f\n' % (key[0], key[1], polyFit[-1], polyFit[0]))
    elif args.p:
        # plot vs log N and record line of best fit data
        bestFitFile = now + '_LoBF-N.txt'
        with open(bestFitFile, 'a') as f:
            f.write('Estimated ln(Error) = k_0 + k_1*ln(N) \n')
            f.write(' n, m, k_0, k_1 \n')
        logAx.set_xlabel(r'Log of number of gridpoints in each dimension, $\log(N)$')
        # plot all sets of data on same axis
        for key in results.keys():
            logAx.plot( np.log(results[key][0]), np.log(results[key][1]), label=r'$n=%d, m=%d$' % (key[0], key[1]) )
            polyFit = np.polyfit(np.log(results[key][0]), np.log(results[key][1]), 1)
            with open(bestFitFile, 'a') as f:
                f.write('%d, %d, %.8f, %.8f\n' % (key[0], key[1], polyFit[-1], polyFit[0]))
    elif args.logH:
        # plot vs log H, don't record best fit data
        logAx.set_xlabel(r'Log of mesh width, $\log(h)$')
        # plot all sets of data on same axis
        for key in results.keys():
            logAx.plot( np.log(1./(results[key][0]-1)), np.log(results[key][1]), label=r'$n=%d, m=%d$' % (key[0], key[1]) )
    else:
        # plot vs log N, don't record best fit data
        logAx.set_xlabel(r'Log of number of gridpoints in each dimension, $\log(N)$')
        # plot all sets of data on same axis
        for key in results.keys():
            logAx.plot( np.log(results[key][0]), np.log(results[key][1]), label=r'$n=%d, m=%d$' % (key[0], key[1]) )
    
    # add legend to plot and save output
    logAx.legend()
    logFig.savefig(now + '_LogError.pdf', bbox_inches='tight')
    # save memory if part of workflow
    plt.close(logFig)

    sys.exit(0)
