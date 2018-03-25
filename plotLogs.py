'''
Functions for plotting log entries

Can be called while, at the end or after a session to visualize
e.g. training progress.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

#TODO: formatting the plot, remove doublin code
#TODO: legend for multiple plots

#if the abscissa is just the numbering of the entries
#@param logs a dictionary with arrays as values
#@param keys is an array of dictornary keys whose values ought to be plotted
    #if None: plot all keys
def plotDefault(destinyFile, logs, keys = None, xLabel = '', yLabel = ''):
    k = keys
    if k is None:
        k = []
        for c in logs:
            k += [c]
    for c in k:
        plt.plot(logs[c])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    fig = plt.gcf()
    fig.set_size_inches(40,10)
    plt.savefig(destinyFile + '.png')
    
    plt.clf()

#if the abscissa has custom values like time samples
#@param logs a dictionary with arrays as values
#@param abcissaKey dictionary key for the abscissa values
#@param keys is an array of dictornary keys whose values ought to be plotted
    #if None: plot all keys except abscissaKey
def plotCustom(destinyFile, logs, abscissaKey, keys = None, xLabel = '', yLabel = ''):
    k = keys
    if k is None:
        k = []
        for c in logs:
            assert np.shape(logs[abscissaKey]) ==  np.shape(logs[c]), "Array of abscissa and ordinate values are not of the same shape!"
            if c != abscissaKey:
                k += [c]
    for c in k:
        plt.plot(logs[abscissaKey], logs[c])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    fig = plt.gcf()
    fig.set_size_inches(40,10)
    plt.savefig(destinyFile + '.png')
    
    plt.clf()
