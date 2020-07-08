# -*- coding: utf-8 -*-

# %% Dependencies

import numpy as np
import pandas as pd

# %%

def principal_drivers_index(df, n=26, px=True, min_assets=0.5):
    """
    Principal Drivers Index
    
    Calculates a correlations index by finding percentage variance explained
    by the principal eigenvector of a correlation matrix
    
    INPUT:
        df - timeseries dataframe of either PX (default) or returns
        n - 26w (default) rolling window
        px - True(default)|False converts to Return series if needed
        min_assets - min percentage of initial universe req to bother calculating
    
    OUTPUT:
        pcr - Principal Drivers Index series
        cum - Cumulative sum of normalised eigenvalues
        cor - correlation matrix used in competition
    """
    
    # require a dataframe of returns
    rtn = df.pct_change() if px == True else df 
    
    # Shrink rtns input where timeseries has rtns for fewer than X% of inputs assets
    rtn = rtn.loc[rtn.notna().sum(axis=1) >= rtn.shape[1] * min_assets, :]
    
    # Dummy Dataframes - use Multi-Index for correlations
    pcr = pd.DataFrame(index=rtn.index[n:], columns=[n])
    cum = pd.DataFrame(index=rtn.index[n:], columns=range(1, rtn.shape[1]+1))
    cor = pd.DataFrame(index=pd.MultiIndex.from_product([rtn.index[n:], rtn.columns]), columns=rtn.columns)
    
    for i, v in enumerate(pcr.index):    # iterate through each date
        
        c = rtn.iloc[i:i+n,:].dropna(axis=1, how='any').corr()    # correlation matrix
        cor.loc[pd.IndexSlice[v, c.columns.tolist()], c.columns.tolist()] = c.values
        
        eig = np.linalg.eigvals(c)                    # calculate eigenvalues
        eig = eig / sum(eig)                          # normalise
        pcr.loc[v] = eig[0]                           # percentage of 1st eigenvalue
        cum.iloc[i, 0:len(eig)] = np.cumsum(eig)      # cumulative sum of eigenvalues
        
    return pcr.astype(float), cum.astype(float), cor.astype(float)

# %% PRINCIPAL DRIVERS INDEX - Class Function

class pdi(object):
    
    def __init__(self):
        return
    
    @property
    def ts(self): return self.__ts
    @ts.setter
    def ts(self, data):              
        if isinstance(data, pd.DataFrame):
            data = data[::-1] if data.index[0] > data.index[-1] else data
            self.__ts = data
        else:
            raise ValueError ('ts takes df object; {} sent'.format(type(data)))
    
    def run(self, n=26, min_assets=0.5):
        """
        """
        x = principal_drivers_index(self.ts, n=n, min_assets=min_assets)
        self.pcr, self.cum, self.cor = x
    
    def plotly_pdi(self, pcr, cum, cor, animations=True):
        """
        Plotly Principal Drivers Index plot with Annimation
        
        Three pane chart
            1. Principal Driver Index (& mean); top half
            2. Cumulative normalised eigenvalues
            3. Correlation matrix at point in time
            
        """
    
        # Chart Schemes
        linewidth = [1]
        cmap = {0:'purple', 1:'turquoise', 2:'grey', 3:'black', 4:'lime', 5:'blue', 6:'orange', 7:'red'}
        colourscale = [[0.0, 'purple'], [0.2, 'darkviolet'], [0.4, 'mediumpurple'], [0.5, 'powderblue'], [0.6, 'paleturquoise'], [0.8, 'turquoise'], [1.0, 'darkturquoise']]
        
        # Static References available throughout chart
        x1, y1 = pcr.index, pcr.iloc[:,0].values
        ymean, ystd = np.mean(y1), np.std(y1)
        c = cor.loc[pd.IndexSlice[pcr.index[-1], :], :].reset_index(level=0, drop=True).iloc[::-1,:]
        
        # Order +1SD, -1SD (with fill area), PDI, PDI_MEAN, Cum. Eigenvealues, Correl Heatmap
        # Order needs to be maintained in the animation
        data=[dict(type='scatter', name='+SD', x=x1, y=[ymean+ystd]*len(x1), xaxis='x1', yaxis='y1', hoverinfo='skip',
                   line=dict(color='whitesmoke', width=linewidth[0], dash='dot')),
              dict(type='scatter', name='-SD', x=x1, y=[ymean-ystd]*len(x1), xaxis='x1', yaxis='y1', hoverinfo='skip', fill='tonexty',
                   line=dict(color='whitesmoke', width=linewidth[0], dash='dot')),
                dict(type='scatter', name='PDI', x=x1, y=y1, xaxis='x1', yaxis='y1',
                   line=dict(color=cmap[0], width=linewidth[0])),
              dict(type='scatter', name='PDI_MEAN', x=x1, y=[ymean]*len(x1), xaxis='x1', yaxis='y1',
                   line=dict(color=cmap[1], width=linewidth[0], dash='dot')),
              dict(type='scatter', name='Cum. Vectors', x=cum.columns, y=cum.iloc[-1,:], xaxis='x2', yaxis='y2',
                  line=dict(color=cmap[1], width=linewidth[0])),
              dict(type='heatmap', name='correl', x=c.index[::-1], y=c.index, z=c.values.tolist(), zmin=-1, zmax=+1,
                   xaxis='x3', yaxis='y3', colorscale=colourscale, showscale=False), ]
    
        layout=dict(title='X-Asset Principal Drivers Index',
                    font = {'family':'Courier New', 'size':12},
                    showlegend = False,
                    margin = {'l':50, 'r':50, 'b':50, 't':50},
                    xaxis1 = {'domain': [0.0, 1], 'anchor': 'y1', 'title': '',},
                    xaxis2 = {'domain': [0, 0.45], 'anchor': 'y2', 'title': '', 'dtick':1, },
                    xaxis3 = {'domain': [0.55, 1], 'anchor': 'y3', 'title': '',},
                    yaxis1 = {'domain': [0.5, 1.0], 'anchor': 'x1', 'title': 'PDI Index', 'hoverformat':'.2f', },
                    yaxis2 = {'domain': [0.0, 0.45], 'anchor': 'x2', 'title': 'cum. eigenvalue', 'hoverformat':'.1%', 'tickformat':'.0%', 'range':[0.1, 1.01]},
                    yaxis3 = {'domain': [0.0, 0.45], 'anchor': 'x3', 'title': '', 'hoverformat':'.1%',},)
        
        fig=dict(data=data, layout=layout)    # Create plotly figure
        
        ### ANIMATION SECTION
        if animations:
            
            # Basic Animation with Slider setup
            frames=[]
            sliders = {'yanchor': 'top', 'xanchor': 'left', 
                       'currentvalue': {'font':{'size': 10}, 'prefix':'Component Vals & Correlation Date: ', 'xanchor': 'left'},
                       'transition': {'duration': 10, 'easing': 'linear'},
                       'pad': {'b': 0, 't': 25}, 'len': 1, 'x': 0, 'y': 0,
                       'steps':[]}
    
            # Iterate through adding data sets to frames for each step in animation
            for i, v in enumerate(cum.index[-1:0:-13].values):
                
                ## This space for calculation required for iteration
                ix = pcr.index.get_loc(v)
                c = cor.loc[pd.IndexSlice[pcr.index[ix], :], :].reset_index(level=0, drop=True).iloc[::-1,:]
                ## END Calculation Space
    
                label = '{}'.format(v)    # string label - used to link slider and frame (data things)
    
                # Append "frame" dictionary to frames list
                frames.append({'name':i, 'layout':{},
                               'data':[dict(type='scatter', y=[ymean+ystd]*len(x1), xaxis='x1', yaxis='y1'),    # +SD
                                       dict(type='scatter', y=[ymean-ystd]*len(x1), xaxis='x1', yaxis='y1'),    # -SD
                                       dict(type='scatter', y=y1, xaxis='x1', yaxis='y1'),                      # PDI
                                       dict(type='scatter', y=[ymean]*len(x1), xaxis='x1', yaxis='y1'),         # PDI_MEAN
                                       dict(type='scatter', y=cum.loc[v,:], xaxis='x2', yaxis='y2'),            # Cum. Eigenvalues
                                       dict(type='heatmap', x=c.index[::-1], y=c.index, z=c.values.tolist(),    # Correl
                                            colorscale=colourscale, showscale=False,xaxis='x3', yaxis='y3'),
                                      ]})
    
                # Append a "step" dictionary to steps list in sliders dict
                sliders['steps'].append({'label':label, 'method': 'animate', 
                                         'args':[[i], {'frame': {'duration': 0, 'easing':'linear', 'redraw': True},
                                                'transition':{'duration': 0, 'easing': 'linear'}}],
                                        })
                
                fig['frames'] = frames
                fig['layout']['sliders'] = [sliders]        # Append completed sliders dictionary to layout
                
        ### END OF ANIMATIONS
        
        return fig