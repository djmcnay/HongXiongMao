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

# Slider on bottom of chart covering whole width
DEFAULT_SLIDERS = {'yanchor':'top', 'xanchor':'left', 
                   'currentvalue': {'font':{'size':12}, 'prefix':'Date: ', 'xanchor':'left'},
                   'transition': {'duration': 500, 'easing': 'linear'},
                   'pad': {'b': 0, 't': 25}, 'len': 1, 'x': 0, 'y': 0,
                   'steps':[]}

# Play & Pause Buttons
DEFAULT_PLAYPAUSE = [{'label': 'Play', 'method': 'animate',
                      'args':[None, {'frame':{'duration':0,'redraw': False},
                                     'fromcurrent': True,
                                     'transition': {'duration':0,'easing':'quadratic-in-out'}}],},
                     {'label': 'Pause', 'method': 'animate',
                      'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                        'mode':'immediate',
                                        'transition': {'duration': 0}}],}]

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

# %%
    
def _quadrants(layout, x=0, y=0, z=100, opacity=0.1,
               colours=['green', 'orange', 'red', 'blue']):
    """
    Quandrants adds 4 quadrant to plot around specified point (default is 0,0)
    """
    
    # Create List of shapes for Plotly Layout & add 4x rectangles
    shapes = [{'type':'rect', # Top Right
               'xref':'x', 'x0':x, 'x1':z,
               'yref':'y', 'y0':y, 'y1':z,
               'line':{'width': 0,},'fillcolor':colours[0], 'opacity':opacity,},
              {'type': 'rect', # Bottom Right
               'xref':'x', 'x0':x, 'x1':z,
               'yref':'y', 'y0':-z, 'y1':y,
               'line':{'width': 0,}, 'fillcolor':colours[1], 'opacity':opacity,},    
              {'type':'rect', # Bottom Left
               'xref':'x', 'x0':-z, 'x1':x, 
               'yref':'y', 'y0':-z, 'y1':y,
               'line':{'width': 0,}, 'fillcolor':colours[2], 'opacity':opacity,},    
              {'type':'rect', # Top Left
               'xref':'x', 'x0':-z, 'x1':x,
               'yref':'y', 'y0':y, 'y1':z, 
               'line':{'width': 0,}, 'fillcolor':colours[3], 'opacity':opacity,},
]
    
    # If 'shapes' not in layout we need to add list
    # otherwise we append each of our rectangles in turn
    if 'shapes' in layout.keys():
        for r in shapes:
            layout['shapes'].append(r)
    else:
        layout['shapes'] = shapes

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

# %%
    
def swirlygram(df, df2=None, n=3, lead=6, trail=18, animation=False,
               quadrants=True, minax=[0, 0], duration=0, transition=500,
               **kwargs):
    """
    Generalised Swirlygram with Animations & Multiple series
    
    Swirlygrams show the absolute level (amplitude) of, for example, a leading
    indicator vs. the change in the indicator over n periods, as well as a tail
    for the most recent past observations.
    
    My original specific case was to show the OECD CLI vs. Normalised GDP (lagged).
    Thus if the CLI is infact leading GDP by 4-8 months the animation should show
    both comets following each other around the plot, circling the origin.
    
    This function is a generalised swirlygram, meaning:
        * Allows multiple series CLI on one chart, using buttons to change
        * Allows an animation of the swirlygram through time
        * Allows for a "reference series" to be animated at the same time
    
    INPUTS:
        df - pd.DataFrame where each column is a seperate series of the CLI and
              the index is a timeseries index
        df2 - (default is None) is a dataframe of a reference series to be plotted
              alongside the CLI in the animation. IMPORTANT NOTE, this MUST have
              the same column headers in the same order as df or bad things happen.
        n - periodic change (default == 3). Currently this is the absolute shift
        lead - shift applied to df2 i.e. lead=6 will move June '18 to December '18
        trail - length of history to show in comet tail (default==18)
        quadrants - True (default) | False and adds coloured quadrants around origin
        animation - False(default) | True depends if we are making an animation
        minax - default = [1, 1], minimum axis width/height
        duration & transition - timings for animation in ms
        **kwargs - mostly for updating layout titles etc... 
    
    NOTES:
        * Absent df2 CLI's will follow normal colourmap
        * Inc df2 all CLIs will be one colour & reference series another
        * chart is forced symmetric around the origin with minimum size of 1
        * inbuilt auto-scaling of x & y. Be careful not to build chart using series
          of very different amplitudes (unless it's deliberate) because the chart will
          size for the largest one and you could lose resolution on the tighter chart
          
    KNOWN PROBLEMS:
        * Plotly animations update layout at the END of the Animation, which 
          means we can either set the range at MAX to begin & zoom at the end
          or PAUSE
    
    """
    
    ### Default Setup
    cmap=COLOURMAP.copy()
    layout=copy.deepcopy(DEFAULT_LAYOUT)
    data, frames=[], []
    sliders = copy.deepcopy(DEFAULT_SLIDERS)
    
    ### Manipulate Data
    x, y = (df - df.shift(n)), df    # x & y df for the cli series 
    
    # Where 2nd DataFrame passed, ensure indices match main df
    # Also shift 2nd dataframe by desired lead & forward fill missing data
    ref = True if isinstance(df2, pd.DataFrame) else False    # Flag about Ref data
    if ref:
        df2 = pd.concat([pd.DataFrame(index=df.index), df2.shift(lead)], axis=1, sort=False).ffill()
        x1, y1 = (df2 - df2.shift(n)), df2
    
    ### Append Data
    # NB/ Colours follow colourmap if NOT Ref, just use cmap[0] and cmap[1] if Ref
    for i, c in enumerate(df.columns):
        # MARKER for most recent observation
        data.append(dict(type='scatter', mode='markers', showlegend=False, hoverinfo='skip',
                    x=[x[c].iloc[-1]], y=[y[c].iloc[-1]],
                    marker=dict(symbol='diamond', color=cmap[0 if ref else i], size=10,
                               line={'color':'black', 'width':1}),))
        # LINE
        data.append(dict(type='scatter', name=c, mode='lines+markers', showlegend=True,
                    x=x[c].iloc[-trail:],
                    y=y[c].iloc[-trail:],
                    line=dict(color=cmap[0 if ref else i], width=1),))
        
        # Reference Dataframe if provided
        if ref:
            # MARKER for most recent observation
            data.append(dict(type='scatter', mode='markers', showlegend=False, hoverinfo='skip',
                        x=[x1[c].iloc[-1]], y=[y1[c].iloc[-1]],
                        marker=dict(symbol='diamond', color=cmap[1], size=10,
                        line={'color':'black', 'width':1}),))
            # LINE
            data.append(dict(type='scatter', name=c, mode='lines+markers', showlegend=True,
                        x=x1[c].iloc[-trail:],
                        y=y1[c].iloc[-trail:],
                        line=dict(color=cmap[1], width=1),))
            
        ## Buttons
        # Only required if > 1 column
        if len(df.columns) > 1:
            if i == 0: l = len(data)                   # find no traces on 1st pass
            visible = [False] * l * len(df.columns)    # List of False = Total No of Traces
            visible[(i*l):(i*l)+l] = [True] * l        # Make traces of ith pass visible

            button= dict(label=c, method='update', args=[{'visible': visible},])
            layout['updatemenus'][0]['buttons'].append(button)     # append the button 
    
    ### Additional Layout Changes

    # Additional Button to make all visible
    if len(df.columns) > 1:
        visible=[True] * l * len(df.columns) 
        button= dict(label='All', method='update', args=[{'visible': visible},])
        layout['updatemenus'][0]['buttons'].append(button)     # append the button 
    
    # Symmetrical Plot around zero
    # Known animation issue that layout updates at end of animation, therefore
    # if ref we use the MAX range to start and shrink at the ANIMATION
    absmax = lambda df, x: np.max([df.abs().max().max(), x]) * 1.05  
    xmax = absmax(x, minax[0]) if ref else absmax(x.iloc[-trail:,:],minax[0])
    ymax = absmax(y, minax[1]) if ref else absmax(y.iloc[-trail:,:],minax[1])
    layout['xaxis']['range'] = [-xmax, xmax]
    layout['yaxis']['range'] = [-ymax, ymax]
    
    # Rectangles of colour for each quadrant
    if quadrants: layout = _quadrants(layout, x=0, y=0)
    
    # Kwargs update to layout
    layout=_update_layout(layout, **kwargs)
    
    ### Create Fig - needs to be done prior to animations
    fig = dict(data=data, layout=layout)
    
    ### Animations
    if animation:
            
        ## Need Play/Pause Buttons
        # Using modified version of plotly example
        playpause=copy.deepcopy(DEFAULT_PLAYPAUSE)
        
        # Append to layout - remembering we may already have a set of buttons in updatemenus
        fig['layout']['updatemenus'].append({'buttons': playpause, 'type': 'buttons', 'showactive': False,
                                             'x':0, 'xanchor':'left', 'y':0, 'yanchor':'bottom',
                                             'direction': 'left', 'pad': {'r': 0, 't': 0},})
        
        ## Build Animation Slider
        
        # Iterate through cli adding data sets to frames for each step in animation
        for i, v in enumerate(x.index.values[trail:-1]):
            
            # Complicated bit is building frame traces for each step
            frame_traces = []
            for j, c in enumerate(df.columns):
                
                frame_traces.append(dict(type='scatter', name='CLI',
                                         x=[x.iloc[i,j]], y=[y.iloc[i,j]]))    # CLI marker
                frame_traces.append(dict(type='scatter', name='CLI',
                                         x=x.iloc[i-trail:i+1, j],             # CLI line
                                         y=y.iloc[i-trail:i+1, j]))
                
                # As above if Reference series added
                if ref:
                    frame_traces.append(dict(type='scatter', name='Ref', x=[x1.iloc[i,j]], y=[y1.iloc[i,j]]))
                    frame_traces.append(dict(type='scatter', name='Ref', x=x1.iloc[i-trail:i+1,j], y=y1.iloc[i-trail:i+1, j]))
                    
            # Append "frame" dictionary to frames list
            # Also update layout to sensible x/y axis for most recent obs
            xmax, ymax = absmax(x.iloc[-trail:,:],minax[0]), absmax(y.iloc[-trail:,:],minax[1])
            frame_layout = {}
            frame_layout={'xaxis':{'range':[-xmax, xmax]}, 'yaxis':{'range':[-ymax, ymax]},}
            frames.append({'name':i, 'layout':frame_layout, 'data':frame_traces})

            # Append a "step" dictionary to steps list in sliders dict
            label = pd.to_datetime(str(v)).strftime('%m/%y')    # String Date Label
            sliders['steps'].append({'label':label, 'method': 'animate', 
                                     'args':[[i], {'frame': {'duration': duration, 'easing':'linear', 'redraw':True},
                                                   'transition':{'duration': transition, 'easing': 'linear'}}],})
            
        fig['frames'] = frames
        fig['layout']['sliders'] = [sliders]        # Append completed sliders dictionary to layout
                
    return fig