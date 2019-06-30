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
import copy

COLOURMAP = {0:'purple', 1:'turquoise', 2:'grey', 3:'black', 4:'lime', 5:'blue', 6:'orange', 7:'red'}

DEFAULT_LAYOUT = dict(title='Title',
                      font={'family':'Courier New', 'size':12},
                      showlegend=True,
                      legend={'orientation':'v'},
                      margin = {'l':75, 'r':50, 'b':50, 't':50},
                      xaxis1= {'anchor': 'y1', 'title': '',},
                      yaxis1= {'anchor': 'x1', 'title': '', 'hoverformat':'.2f', 'tickformat':'.1f'},
                      updatemenus= [dict(type='buttons',
                                         active=-1, showactive = True,
                                         direction='right',
                                         y=0.25, x=1.1,
                                         pad = {'l':0, 'r':0, 't':0, 'b':0},
                                         buttons=[])],
                      annotations=[],)

# %%

def _update_layout(layout, **kwargs):
    """
    Helper function to update layout dictionary from kwargs passed
    
    Bespoke Key/Value pairs:
        - WRITE SOME PAIRS HERE 
    
    Limitations - building this for all possible combinations would make it almost
    as complicated as manually building the layout dictionary by hand, so there are
    some known limitations.
        * Only updates 1st array of buttons
    
    """
 
    ### First Level Stuff - test if kwarg is a key in layout
    for k, v in kwargs.items():
        
        if k in layout and isinstance(v, str):
            layout[k] = v    # where v is str, do straight replacement
        
        elif k in layout and isinstance(v, dict):
            # if v is a dict, either replace or append to dict
            for k1, v1, in v.items():
                if v1 in layout[k][k1]:
                    layout[k][k1] = v1
    
    ### Second Level => bespoke kwarg inputs
    
    # Button position update
    if 'button_position' in kwargs:
        k, v = 'button_position', kwargs['button_position']
        layout['updatemenus'][0]['x'] = v[0]
        layout['updatemenus'][0]['y'] = v[1]
        if len(v) > 2:
            layout['updatemenus'][0]['pad']={'l':v[2],'r':v[3],'t':v[4],'b':v[5]}
    
    # Button Direction
    if 'button_direction' in kwargs:
        layout['updatemenus'][0]['direction'] = kwargs['button_direction'] 
    
    return layout

# %% Scatter Plot from Dataframe with Group Buttons

def scatter_from_dataframe(df, **kwargs):
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
    
    # Bring in Defaults from Global
    cmap=COLOURMAP
    layout=copy.deepcopy(DEFAULT_LAYOUT)
    
    ## Basic Layout Dictionary
    # Needs to come first or buttons have nowhere to append too
    layout= _update_layout(layout, **kwargs)
    
    ## Iterate Through Each column of Dataframe to append
    data = []
    for i, c in enumerate(df): 
        data.append(dict(type='scatter', name=c, x=df.index, y=df.loc[:,c],
                         line={'color':cmap[i], 'width':1}))

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

    fig = dict(data=data, layout=layout)

    return fig