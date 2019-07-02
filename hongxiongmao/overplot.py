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

COLOURMAP = {0:'purple', 1:'turquoise', 2:'grey', 3:'black', 4:'green',
             5:'blue', 6:'crimson', 7:'orange', 8:'mediumvioletred'}

DEFAULT_LAYOUT = dict(title='Title',
                      font={'family':'Courier New', 'size':12},
                      showlegend=True,
                      legend={'orientation':'v'},
                      margin = {'l':75, 'r':50, 'b':50, 't':50},
                      xaxis= {'anchor': 'y1', 'title': '',},
                      yaxis= {'anchor': 'x1', 'title': '', 'hoverformat':'.2f', 'tickformat':'.1f'},
                      updatemenus= [dict(type='buttons',
                                         active=-1, showactive = True,
                                         direction='down',
                                         y=0.5, x=1.1,
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
    cmap= COLOURMAP.copy()
    layout= copy.deepcopy(DEFAULT_LAYOUT)  # deepcopy important bedded dicts
    
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

# %% Swirlygrams
    
def swirlygram(df, n=3, trail=18, quadrants=True, **kwargs):
    """
    Static Swirlygram with multiple traces
    
    Shows the absolute level of (for example) a composite leading indicator and 
    the change in the indictor over n periods, with a for previous observations.
    
    This version is STATIC and only shows the most recent Swirlygram for each
    column in the input dataframe, with buttons to shrink the number of plots 
    shown; seperate function being built for an animated timeseries Swirlygram.
    
    INPUTS:
        df - dataframe with each column being a seperate series
        n - periodic change (default == 3)
             Currently this is the absolute shift, but will add percentage
        trail - length of history to show in comet tail (default==18)
        quadrants - True (default) | False and adds coloured quadrants around origin
        
    
    DEVELOPMENTS:
        * qudrant function to allow us to move origin about
        * Percentage change or other x-axis stuff
            
    """
    
    ### Default Setup
    cmap=COLOURMAP.copy()
    layout=copy.deepcopy(DEFAULT_LAYOUT)
    layout=_update_layout(layout, **kwargs)
    
    ### Manipulate Data
    df = df.iloc[-(trail+n):,:]               # subset input df to required length
    x = (df - df.shift(n)).iloc[-trail:,:]    # x-axis given as change
    y = df.iloc[-trail:,:]                      # y-axis suset to length of tail
    
    ### Append Data
    data=[]
    for i, c in enumerate(df.columns):

        # LINE
        data.append(dict(type='scatter', name=c, mode='lines+markers', showlegend=True,
                    x=x.loc[:,c], y=y.loc[:,c],
                    line=dict(color=cmap[i], width=1),))
            
        # MARKER for most recent observation
        data.append(dict(type='scatter', mode='markers', showlegend=False, 
                    x=[x.iloc[-1,:].loc[c]], y=[y.iloc[-1,:].loc[c]],
                    marker=dict(symbol='diamond', color=cmap[i], size=10,
                               line={'color':'black', 'width':1}),))
    
        ## Buttons
        if i == 0: l = len(data)                   # find no traces on 1st pass
        visible = [False] * l * len(df.columns)    # List of False = Total No of Traces
        visible[(i*l):(i*l)+l] = [True] * l        # Make traces of ith pass visible
        
        button= dict(label=c, method='update', args=[{'visible': visible},])
        layout['updatemenus'][0]['buttons'].append(button)     # append the button 
    
    ### Additional Layout Changes
    
    # Additional Button to make all visible
    visible=[True] * l * len(df.columns) 
    button= dict(label='All', method='update', args=[{'visible': visible},])
    layout['updatemenus'][0]['buttons'].append(button)     # append the button 
    
    # Symmetrical Plot around zero
    absmax = lambda df, x=1: np.max([df.abs().max().max(), x]) * 1.05    
    layout['xaxis']['range'] = [-absmax(x), absmax(x)]
    layout['yaxis']['range'] = [-absmax(y), absmax(y)]
    
    # Rectangles of colour for each quadrant
    if quadrants:
        shapes = [{'type':'rect', # Top Right
                   'xref':'x', 'x0':0, 'x1':100,
                   'yref':'y', 'y0':0, 'y1':100,
                   'line':{'width': 0,},'fillcolor':'green', 'opacity': 0.1,},    
                  {'type':'rect', # Bottom Left
                   'xref':'x', 'x0':-100, 'x1':0, 
                   'yref':'y', 'y0':-100, 'y1':0,
                   'line':{'width': 0,}, 'fillcolor':'red', 'opacity': 0.1,},    
                  {'type':'rect', # Top Left
                   'xref':'x', 'x0':-100, 'x1':0,
                   'yref':'y', 'y0':0, 'y1':100, 
                   'line':{'width': 0,}, 'fillcolor':'blue', 'opacity': 0.1,},
                  {'type': 'rect', # Bottom Right
                   'xref':'x', 'x0':0, 'x1':100,
                   'yref':'y', 'y0':-100, 'y1':0,
                   'line':{'width': 0,}, 'fillcolor':'orange', 'opacity': 0.1,},]
        layout['shapes'] = shapes
    
    return dict(data=data, layout=layout)
