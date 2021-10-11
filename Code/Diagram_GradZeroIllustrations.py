#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:44:10 2021

@author: will
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex='True')

def AxisSettings(ax, edgeWidth):
	ax.set_xlim([0.,1.])
	ax.set_ylim([-edgeWidth/2,1.+edgeWidth/2])
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	ax.set_aspect('equal')
	return

def f1(x,y):
	return np.sin( np.pi*(x-0.5) ) * np.sin( np.pi*(y-0.5) )

def f2(x,y):
	return -(0.5-y)

def f3(x,y):
	return (y-0.5)*np.sin( (4*(x))**2 + (8*(y-0.5))**2 )
	#return np.sin( np.pi*(y-0.5) ) * ( np.cos( np.pi*(x-0.5) ) )**2

def EmphEdge(ax, edgeWidth):
	ax.plot([0.,1.], [0.5-edgeWidth/2, 0.5-edgeWidth/2], '-r')
	ax.plot([0.,1.], [0.5+edgeWidth/2, 0.5+edgeWidth/2], '-r')

def ThickenEdge(x, z, edgeWidth, nPts=100, normalise=False, forceZero=True):
	zShape = z.shape
	zHalf = zShape[0]//2
	zThick = np.zeros((zShape[0]+nPts, zShape[1]), dtype=float)
	xLong = np.zeros((zShape[0]+nPts), dtype=float)
	
	xLong[:zHalf] = x[:zHalf] - 0.5*edgeWidth
	xLong[zHalf:zHalf+nPts] = np.linspace(x[zHalf]-edgeWidth/2, x[zHalf]+edgeWidth/2, nPts)
	xLong[zHalf+nPts:] = x[zHalf:] + 0.5*edgeWidth
	
	zThick[:zHalf,:] = np.copy(z[:zHalf,:])
	if forceZero:
		zThick[zHalf:zHalf+nPts,:] = 0.
	else:
		zThick[zHalf:zHalf+nPts,:] = np.copy(z[zHalf,:])
	zThick[zHalf+nPts:,:] = np.copy(z[zHalf:,:])
	
	if normalise:
		zThick = zThick/np.max(np.abs(zThick))
	
	return xLong, zThick

nPts = 1000
edgeWidth = 0.1
x = np.linspace(0.,1.,nPts)
X, Y = np.meshgrid(x,x)

xLong, Z1 = ThickenEdge(x, f1(X,Y), edgeWidth)
Z2 = ThickenEdge(x, f2(X,Y), edgeWidth)[1]
Z3 = ThickenEdge(x, f3(X,Y), edgeWidth)[1]

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
for a in ax:
	AxisSettings(a, edgeWidth)
	EmphEdge(a, edgeWidth)

# create contour plots of the various functions
cPlots = [None,None,None]
cPlots[0] = ax[0].contourf(x, xLong, Z1, 25)
cPlots[0] = ax[0].contourf(x, xLong, Z1, 25)
cPlots[1] = ax[1].contourf(x, xLong, Z2, 25)
cPlots[1] = ax[1].contourf(x, xLong, Z2, 25)
cPlots[2] = ax[2].contourf(x, xLong, Z3, 25)
cPlots[2] = ax[2].contourf(x, xLong, Z3, 25)

# colourbar settings
p0 = ax[0].get_position().get_points().flatten()
p2 = ax[2].get_position().get_points().flatten()
ax_cbar = fig.add_axes([p0[0], 0.25, p2[2]-p0[0], 0.05])
cbar = fig.colorbar(cPlots[0], cax=ax_cbar, orientation='horizontal')
cbar.set_ticks(np.linspace(-1.,1.,num=9))

# save and show
fig.savefig('../Diagrams/Numerical_Results/Diagram_GradZeroIllustrations.pdf', bbox_inches='tight')
fig.show()

#def F1(x,y):
#	return (y-0.5)*np.sin( (4*(x))**2 + (8*(y-0.5))**2 )
#
#fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharey=True)
#for a in ax2:
#	AxisSettings(a, edgeWidth)
#	EmphEdge(a, edgeWidth)
#	
#zData = ThickenEdge(x, F1(X,Y), edgeWidth)[1]
#conPlot2 = ax2[0].contourf(x, xLong, zData, 25)
#ax2[1].contourf(x, xLong, zData * (Z2 + 1.), 25)
#
#p20 = ax2[0].get_position().get_points().flatten()
#p21 = ax2[1].get_position().get_points().flatten()
#ax_cbar2 = fig2.add_axes([p20[0], 0.1, p21[2]-p20[0], 0.05])
#cbar2 = fig2.colorbar(conPlot2, cax=ax_cbar2, orientation='horizontal')
#cbar2.set_ticks(np.linspace(-1.,1.,num=9))
#
#fig2.show()