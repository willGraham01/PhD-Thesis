#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:19:15 2021

@author: will
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, rc
rc('text', usetex=True) 
from mpl_toolkits.mplot3d import Axes3D  

def AxesSettings(ax):
	ax.set_xlabel(r'$x_1$')
	ax.set_ylabel(r'$x_2$')
	ax.set_xlim([0.,1.])
	ax.set_ylim([0.,1.])
	ax.set_aspect('equal')
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	return

def f1(x,y):
	return np.sin( np.pi*(y-0.5) ) * np.sin( np.pi*(x-0.5) )

def f2(x,y):
	return (y-0.5)#(8/3)*(y-0.5)**3 + 0.571429*(y-0.5)**2 - (5/3)*(y-0.5)

def f3(x,y):
	return np.sin( np.pi*(y-0.5) ) * (np.cos( np.pi*(x-0.5) ))**2

def EmphEdge(ax, edgeWidth):
	ax.plot([0.,1.], [0.5-edgeWidth/2,0.5-edgeWidth/2], 'r')
	ax.plot([0.,1.], [0.5+edgeWidth/2,0.5+edgeWidth/2], 'r')
	ax.plot([0.,0.], [0.5-edgeWidth/2,0.5-edgeWidth/2], 'r')
	ax.plot([0.,0.], [0.5+edgeWidth/2,0.5+edgeWidth/2], 'r')
	return

def Normalise(zData):
	return zData/np.max(np.abs(zData))

def ThickenArray(Z, x, edgeWidth, nPts=10, normalise=True):
	shapeZ = Z.shape
	z = np.zeros((shapeZ[0]+nPts, shapeZ[1]), dtype=float)
	xPts = x.shape[0]
	xData = np.copy(Z[xPts//2,:])
	
	z[:(xPts//2),:] = np.copy(Z[:(xPts//2),:])
	z[xPts//2 + nPts:,:] = np.copy(Z[xPts//2:,:])
	for i in range(nPts):
		z[xPts//2+i,:] = xData
	
	xLonger = np.zeros((shapeZ[0]+nPts,), dtype=float)
	xLonger[:xPts//2] = np.copy(x[:xPts//2]) - edgeWidth/2
	xLonger[(xPts//2):(nPts + xPts//2)] = np.linspace(x[(xPts//2)] - edgeWidth/2 ,x[(xPts//2)] + edgeWidth/2, num=nPts)
	xLonger[xPts//2+nPts:] = np.copy(x[xPts//2:]) + edgeWidth/2
	
	if normalise:
		z = Normalise(z)
	return z, xLonger

edgeWidth = 0.1
nPts = 1000
x = np.linspace(0.,1.,nPts)
X, Y = np.meshgrid(x, x)
Z1 = f1(X,Y)
Z2 = f2(X,Y)
Z3 = f3(X,Y)

z1, xLonger = ThickenArray(Z1, x, edgeWidth, nPts=100, normalise=True)
z2 = ThickenArray(Z2, x, edgeWidth, nPts=100, normalise=True)[0]
z3 = ThickenArray(Z3, x, edgeWidth, nPts=100, normalise=True)[0]

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
# Turn all axes off for manual labelling
for a in ax:
	AxesSettings(a)
	EmphEdge(a, edgeWidth)

# plot the functions....
#colorMap = 'gnuplot'
#colorMap = 'plasma'
colorMap = 'viridis'
conPlot1 = ax[0].contourf(x, xLonger, z1, 25, cmap=colorMap)
ax[1].contourf(x, xLonger, z2, 25, cmap=colorMap)
ax[2].contourf(x, xLonger, z3, 25, cmap=colorMap)

# colorbar hacks
p0 = ax[0].get_position().get_points().flatten()
p2 = ax[2].get_position().get_points().flatten()
ax_cbar = fig.add_axes([p0[0], 0.25, p2[2]-p0[0], 0.05])
cbar = fig.colorbar(conPlot1, cax=ax_cbar, orientation='horizontal')
cbar.set_ticks(np.linspace(-1,1,9))

#figure title

# show figure, and save if desired
fig.show()
saveStr = 'GradZeroProfiles.pdf'
fig.savefig(saveStr, bbox_inches='tight')















#from mayavi import mlab
#from matplotlib.cm import get_cmap # for viridis
#
#def f1(x, y):
#    return np.cos( np.pi*(x-0.5) ) * np.sin( -np.pi*(y-0.5) )
#
## data for the surface
#nPts = 100
#x = np.linspace(0., 1., nPts)
#X, Y = np.meshgrid(x, x)
#Z = f1(X, Y)
#
## data for the scatter
#xx = x
#yy = 0.5*np.ones_like(x)
#zz = np.zeros_like(x)
#
#fig = mlab.figure(bgcolor=(1,1,1))
## note the transpose in surf due to different conventions compared to meshgrid
#su = mlab.surf(X.T, Y.T, Z.T)
#sc = mlab.points3d(xx, yy, zz, scale_factor=0.05, scale_mode='none',
#                   opacity=1.0, resolution=20, color=(1,1,1))
#
##mlab.axes(xlabel=r'$x_1$', ylabel=r'$x_2$', zlabel=r'$u(x)$', x_axis_visibility=True)
#
## manually set viridis for the surface
#cmap_name = 'viridis'
#cdat = np.array(get_cmap(cmap_name,256).colors)
#cdat = (cdat*255).astype(int)
#su.module_manager.scalar_lut_manager.lut.table = cdat
#
#mlab.show()