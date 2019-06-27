#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:20:28 2019

@author: David
"""
import pickle
import numpy as np
from download import alphavantage as av
import utilities
#import badgermodels as bm
from principaldrivers import pdi

# %%

#dl = av()
#x = dl.dl_from_ticker_dict(dl.us_etfs, output_size='full')
#filename='data_alphavantage_daily'
#a = utilities.picklemerger(filename=filename, b=x, blend='left',
#                           path='data', create_new=True, output=True)

# %%

def dendrogram_duo_mpl(rtn, n=[260, 13]):

    # Extra dependencies
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette, cophenet
    from scipy.spatial.distance import pdist
    
    # Setup figure
    fig = plt.figure(figsize=(7.5, 3.5), dpi=125)
    plt.tight_layout()
    fig.suptitle('Dendrogram of weekly Correlations', fontsize=12)
    plt.rcParams['font.family'] = "Courier New"
    plt.rcParams.fontsize = 8

    colourscale = [[0.0, 'purple'], [0.2, 'darkviolet'], [0.4, 'mediumpurple'], [0.5, 'powderblue'],
                   [0.6, 'paleturquoise'], [0.8, 'turquoise'], [1.0, 'darkturquoise']]
    set_link_color_palette(list(zip(* colourscale))[1])    # update colourscale for dendrogram

    # LHS dendrogram
    rtn0 = rtn.copy().dropna(how='any') if n[0] == 'max' else rtn.iloc[-n[0]:,:].dropna(how='any')
    mat0 = rtn0.corr()                          # correlation matrix
    ax0 = fig.add_axes([0, 0, 0.49, 0.85])      # append axes to figure
    Z0 = linkage(mat0, 'complete')              # calculate linkages
    dendrogram(Z0, orientation='right', labels=mat0.columns, leaf_font_size=8)
    c0, _ = cophenet(Z0, pdist(mat0))           # cophenet correl coeff

    # RHS dendrogram
    rtn1 = rtn.iloc[-n[1]:,:].dropna(how='any').copy()
    mat1 = rtn1.corr()                          # correlation matrix
    ax1 = fig.add_axes([0.51, 0, 0.49, 0.85])   # append axes to figure
    Z1 = linkage(mat1, 'complete')              # calculate linkages
    dendrogram(Z1, orientation='left', labels=mat1.columns, leaf_font_size=8)
    c1, _ = cophenet(Z1, pdist(mat1))           # cophenet correl coeff

    # Update axiswise stuff
    ax0.set_title('{} to {}'.format(rtn0.index[0], rtn0.index[-1]), fontsize=8)
    ax1.set_title('{} weeks'.format(n[1]), fontsize=8)
    for x in [ax0, ax1]:
        x.tick_params(axis='x', labelsize=8)
        x.set_xlabel('distance')
        x.grid(True, linewidth=0.5, color='lightgrey')
        for s in ['left','bottom','right','top']:
            x.spines[s].set_visible(False)

    # Match x-axis ticks
    xliml, xlimr = ax0.get_xlim(), ax1.get_xlim()
    if np.max(xliml) > np.max(xlimr):
        ax1.set_xlim(xliml[::-1])
    else:
        ax0.set_xlim(xlimr[::-1])

    # Add Cophenet correlation coefficients
    ax0.text(np.max(ax0.get_xlim())*0.9, ax0.get_ylim()[1]*0.9, 'c = {:.2f}'.format(c0), ha='right')
    ax1.text(np.max(ax1.get_xlim())*0.9, ax1.get_ylim()[1]*0.9, 'c = {:.2f}'.format(c1), ha='left')

    return fig

# %%

infile = open('data/data_alphavantage_daily', 'rb')
a = pickle.load(infile)
infile.close()

etfs=['SPY', 'VEA', 'VWO', 'GOVT', 'TIP', 'LQD', 'HYG', 'EMB', 'EMLC', 'GLD']
px = utilities.daily2weekly(a['close']).loc[:,etfs]

# %%

dendrogram_duo_mpl(px.pct_change())