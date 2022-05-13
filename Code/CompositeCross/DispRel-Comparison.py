#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:03:15 2022

@author: will

Script to compare the dispersion relations computed by the VP and FDM methods.

Since the methods might compute DR values at different \qm points, 
and for a differing number of bands, we are required to interpolate the available
data and then compare both DRs at a selection of common points.
"""

import argparse
import sys

import numpy as np
from numpy import pi
from numpy.random import random
from scipy.interpolate import griddata as interp 
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from CompMes_FDMAnalysis import LoadAllFromKey as Load_FDM
from CompMes_FDMAnalysis import GetBand as GetBand_FDM

from CompMes_VarProbAnalysis import LoadAllFromKey as Load_VP
from CompMes_VarProbAnalysis import GetBand as GetBand_VP

# for the purposes of testing, fix the filenames that we want to compare between
#f_FDM = './FDM_Results/nPts51-N251-10evals.csv' # 251 gridpoints, 51 qm values in each component, 10 evs
#f_VP = './VP_Results/nPts25-global-N5-M12-t1loops0-24.csv' # M=12 modes, 25 qm values in each component, 5 evs

#Command-line execution
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Script to compare the dispersion relations computed by the VP and FDM methods: result is printed to the terminal! Can produce figures on request.')
    parser.add_argument('f_FDM', nargs=1, type=str, help='<Required> Output data file from FDM solve.')
    parser.add_argument('f_VP', nargs=1,type=str,  help='<Required> Output data file from VP solve.')
    parser.add_argument('-N', default=100000, type=int, help='<Default 1e5> Number of interpolant points to use.')
    parser.add_argument('-u', action='store_true', help='Use uniform interpolants (via a grid of size sqrt(N)*sqrt(N)) over random points.')
    parser.add_argument('-p', action='store_false', help='Do not create heatmaps of the difference between the DRs.')
    parser.add_argument('-fOut', default='./', type=str, help='<Default .> File location to save plot outputs to, ignored when -p not passed.')

    args = parser.parse_args()
    print(args)
    
    # load eigenvalues
    e_FDM, _ = Load_FDM(args.f_FDM[0], funsToo=False)
    e_VP, _ = Load_VP(args.f_VP[0], funsToo=False)
    # this is how many eigenvalues we have for each method
    nEvalsFDM = e_FDM.shape[1] - 2
    nEvalsVP = e_VP.shape[1] -  3
    # we can only work with bands that both methods gave data for
    nEvals = np.min([nEvalsFDM, nEvalsVP])
    # number of data points we need
    
    # put into bands
    FDM_bands = []
    VP_bands = []
    for n in range(nEvals):
        VP_bands.append(GetBand_VP(n+1, e_VP, removeFailed=True))
        FDM_bands.append(GetBand_FDM(n+1, e_FDM, tol=1e-5))
        
    if args.u:
        N = int(np.floor(np.sqrt(args.N)))
        tRange = np.linspace(-pi, pi, endpoint=True, num=N)
        thetaSamples = np.stack(np.meshgrid(tRange, tRange), -1).reshape(-1,2)
    else:
        # random interpolants to be used   
        N = args.N
        thetaSamples = 2*pi * random((N,2)) - pi
    
    # iVals_[i,j] = interpolant for thetaSamples[i] within band j
    iVals_VP = np.zeros((N,nEvals),dtype=float)
    iVals_FDM = np.zeros((N,nEvals),dtype=float)
    
    for b in range(nEvals):
        # for band b, let us compare the two dispersion relations over the sample points
        iVals_VP[:,b] = interp(VP_bands[b][:,0:2], VP_bands[b][:,2], thetaSamples, method='cubic')
        iVals_FDM[:,b] = interp(FDM_bands[b][:,0:2], FDM_bands[b][:,2], thetaSamples, method='cubic')
    
    diffs = iVals_VP - iVals_FDM
    absDiffs = np.abs(diffs)
    maxDiffs = np.max(absDiffs, axis=0)
    
    print('Absolute value differences (in ascending band order):')
    print(maxDiffs)
    
    # create plots if required
    if args.p:
        for b in range(nEvals):
            saveStr = args.fOut + 'DRcompare-band-%d.pdf' % b
            
            fig, ax = plt.subplots(1)
            ax.set_aspect('equal')
            ax.set_xlabel(r'$\frac{\theta_1}{\pi}$')
            ax.set_ylabel(r'$\frac{\theta_2}{\pi}$')
            #ax.set_title(r'Dispersion Relation difference (VP-FDM), band $%d$' % b)
            #cont = ax.tricontourf(thetaSamples[:,0]/pi, thetaSamples[:,1]/pi, diffs[:,b])
            ax.set_title(r'(Abs) Dispersion Relation difference, band $%d$' % b)
            cont = ax.tricontourf(thetaSamples[:,0]/pi, thetaSamples[:,1]/pi, absDiffs[:,b])
            fig.colorbar(cont)
            fig.savefig(saveStr)
    
    sys.exit(0)