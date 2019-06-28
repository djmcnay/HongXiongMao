#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overplot is a high level overlay for Plotly

It's purpose is similar in function to Cufflinks or Plotly_Express & may even
use those packages to help build charts. Generally these plots are tailored to
weird charts I wanted to build, but could be repeated.


"""

import numpy as np
import pandas as pd

# %% Scatter Plot from Dataframe with Group Buttons

def scatter_from_dataframe(df, title='', ytitle='', **kwargs):
    """
    Plotly Timeseries Scatter Chart - with Group
    
    Simply Plotly Scatter plot with multiple traces (lines).
    Takes pd.Dataframe & creates chart with each column as trace
    Assumption is that xaxis will be the date axis.
    
    Option to add button groups. Input is dictionary of the form
        {'NAME':['Col1', Col2]}
        Where dict key will be the button name &
        List will be the df columns shown   
    """
    
    # Colourscheme
    cmap = {0:'purple', 1:'turquoise', 2:'grey', 3:'black', 4:'lime', 5:'blue', 6:'orange', 7:'red'}
    
    ## Basic Layout Dictionary
    # Needs to come first or buttons have nowhere to append too
    layout=dict(title=title, font = {'family':'Courier New', 'size':12},
                showlegend = True, legend=dict(orientation="h"),
                margin = {'l':75, 'r':50, 'b':50, 't':50},
                yaxis={'title':ytitle, 'hoverformat':'.2f', 'tickformat':'.2f',},
                updatemenus= [dict(type='buttons', active=0, showactive = True,
                                         direction='right', pad = {'l':0, 'r':35, 't':0, 'b':0},
                                         y=-0.1, x=1,
                                         buttons=[])],
               )
    
    ## Iterate Through Each column of Dataframe to append
    data = []
    for i, c in enumerate(df): 
        data.append(dict(type='scatter', name=c, x=df.index, y=df.loc[:,c], line={'color':cmap[i], 'width':1}))

    ## Add Buttons
    # Check if a dictionary of groups was passed to kwargs
    # Should be of the form {'NAME':[LIST of COLUMN VARNAMES]}
    if 'groups' in kwargs:
        for g in kwargs['groups']:
            
            # Find Boolean index of Varnames in Dataframe Column Varnames
            # Use this to adjust Visible/Hidden on data traces
            visible = np.in1d(df.columns, kwargs['groups'][g])
            button= dict(label=g, method='update', args=[{'visible': visible}])
            layout['updatemenus'][0]['buttons'].append(button)    # append the button 

    return dict(data=data, layout=layout)