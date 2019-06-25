#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POKEMON Fair Value Bond Model
    
Model estimates Fair Value or a long (10-year) bond using multi-variate
regression. We assume the long term yield can be expressed as a combination
of the short term interest rate, inflation and growth expectations.

    FV = Intercept + (B1 * short_rate) + (B2 * inflation)
                   + (B3 * growth) + (B4 * global factor)

Where possible we want the model to be forward looking:
    
    interest rate - proxied by the 2-year nominal where possible
    inflation - ideally swap or survey based i.e. UMich or 5y5y fwd
    growth - PMI, Tankan or similar

Default here is to download data from Quandl for a specified region and run
both a full_sample & a rolling regression. The model metrics are stored in 
dictionaries is the "Default Data & Attributes" cell. These aren't necessarily
the unconstrained variables I would choose, but are the best compromise I 
can find whilst limiting to Quandl datasets.

CLASS MODULES:
    pokemon - acts on a single region or set of data
    pokemon_go - combines multiple regions
    
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Homemade Dependencies
from hongxiongmao.utilities import tools, download

# %%

class pokemon(object):
    """
    Pokemon
    
    For more complete information on the model read the help on the Class Module.
    This class operates on a single region (or set of input data). 
    
    ATTRIBUTES:
        self.data = model input data; can be input manually or via dl function
        self.stats_full = full period statistics (pretty useless)
        self.stats_roll = rolling window statistics
        self.pokemodel = DataFrame with model output
    
    FUNCTIONS:
        download_and_run()
        pull_data_from_quandl()
        model()
    """
    
    def __init__(self, region='US'):
        self.region = region
        return
    
    @property
    def data(self): return self.__data
    @data.setter
    def data(self, data):              
        if isinstance(data, pd.DataFrame):
            data = data[::-1] if data.index[0] > data.index[-1] else data
        else:
            raise ValueError ('data takes pd.Dataframe object; {} sent'.format(type(data)))
        self.__data = data
    
    # %% Ticker Dictionaries for Quandl Data
    
    quandl_us = {'US10YR':{'tickers':'USTREASURY/YIELD','fields':['10 YR']},
                 'US02YR':{'tickers':'USTREASURY/YIELD','fields':['2 YR']},
                 'UMich':{'tickers':'UMICH/SOC33','fields':['Median']},
                 'ISM':{'tickers':'ISM/MAN_PMI', 'fields':[]},
                 'REER':{'tickers':'BIS/EM_MRBUS','fields':[]}
                 }

    quandl_bund = {'DR10YR':{'tickers':'BUNDESBANK/BBK01_WT1010','fields':[]},
                   'DR02YR':{'tickers':'BUNDESBANK/BBK01_WT0202','fields':[]},
                   'CPI':{'tickers':'RATEINF/INFLATION_DEU','fields':[]},
                   'OECD_CLI':{'tickers':'OECD/MEI_CLI_LOLITOAA_DEU_M', 'fields':[]},
                   'REER':{'tickers':'BIS/EM_MRBXM','fields':[]}
                   }

    quandl_gilt = {'UK10YR':{'tickers':'BOE/IUDMNZC','fields':[]},
                   'UK3MLIBOR':{'tickers':'ECB/FM_M_GB_GBP_RT_MM_GBP3MFSR__HSTA','fields':[]},
                   'CPI':{'tickers':'RATEINF/INFLATION_GBR','fields':[]},
                   'OECD_CLI':{'tickers':'OECD/MEI_CLI_LOLITONO_GBR_M', 'fields':[]},
                   'REER':{'tickers':'BIS/EM_MRBGB','fields':[]}
                   }

    quandl_jgb = {'JP10YR':{'tickers':'MOFJ/INTEREST_RATE_JAPAN_10Y','fields':[]},
                  'JP02YR':{'tickers':'MOFJ/INTEREST_RATE_JAPAN_2Y','fields':[]},
                  'CPI':{'tickers':'RATEINF/INFLATION_JPN','fields':[]},
                  'OECD_CLI':{'tickers':'OECD/MEI_CLI_LOLITOAA_JPN_M', 'fields':[]},
                  'REER':{'tickers':'BIS/EM_MRBJP','fields':[]}
                   }
    
    quandl_dicts = {'US':quandl_us, 'DR':quandl_bund, 'UK':quandl_gilt, 'JP':quandl_jgb}
    
    # %%
    
    def download_and_run(self, region=None, start_date='10y', rolling_window=36):
        """
        For single region pull Quandl data and run Pokemon model
        """
        region = self.region if region is None else region
        self.pull_data_from_quandl(region=region, start_date=start_date)
        self.model(window=rolling_window)
        return
    
    # %% Quandl Data Import
    def pull_data_from_quandl(self, region='US', start_date='5y', freq='monthly', output=False):
        """
        Download Pokemon Data from Quandl
        """
        
        # Select relevant dictionary from regional input
        if region in list(self.quandl_dicts.keys()):
            tick_dic = self.quandl_dicts[region]
        else:
            raise ValueError('region {} doesn\'t currently have quandl tickers for Pokemon model'
                             .format(region))
        
        # Iterate through dictionary & pull data
        for i, v in enumerate(tick_dic.items()):
            j = tick_dic[v[0]]
            x = download().quandl_ts(tickers=j['tickers'], fields=j['fields'],
                                  start_date=start_date, freq=freq)
            
            x.columns = [v[0]]    # Update column header with dictionary key
            
            # Merge new Dataframe output df
            if i == 0: data = x
            else: data = tools().df_merger(data, x)
        
        # output determines if we return data or set attribute
        if output: return data
        else: self.data = data
    
    # %%
    def model(self, df=None, window=36, valiadate_data=True, output=False):
        """
        Pokemon Fair Value Model
        
        INPUT:
            df - df with long-bond in col0; regression parameters in cols 1+
            window - 36(default) rolling period window for regression
            validate_date - True(default)|False forward fill missing data
        
        OUTPUT:
            self.pokemodel - model output
            self.stats_full - regression stats for full period data
            self.stats_roll - same but rolling
        """
        
        # Use internal data attribute if None is provided
        df = self.data if df is None else df
        
        if valiadate_data is True:
            df.fillna(method='ffill', inplace=True)   # Forward fill missing data
            df = df[~df.isna().any(axis=1)]           # Each row complete
        
        # Dummy Dataframe(s)
        guess = pd.DataFrame(data=df.iloc[:,0])
        guess.columns = ['long_bond']
        
        coeff_vn = ['r2', 'intercept']
        coeff_vn.extend(list(df.columns.values)[1:])
        stats = pd.DataFrame(columns=coeff_vn)
        
        # Full Sample Period
        y, X = df.iloc[:,0], df.iloc[:,1:]
        lm = LinearRegression().fit(X,y)
        guess['full_sample'] = lm.predict(X)
        stats.loc['full_sample',['r2','intercept']]=lm.score(X, y),lm.intercept_
        stats.loc['full_sample'].iloc[2:] = lm.coef_
        
        # Rolling Window
        for i, v in enumerate(guess.index):
            if i < window:
                continue
            
            y = df.iloc[i-window:i,0]     # Dependant Var [long bond]
            X = df.iloc[i-window:i,1:]    # Independant [short, inf, growth]
            roll_lm = LinearRegression().fit(X,y)
            guess.loc[v,'rolling'] = roll_lm.predict(X)[-1]
            stats.loc[v,['r2', 'intercept']] = roll_lm.score(X, y), lm.intercept_
            stats.loc[v].iloc[2:] = roll_lm.coef_ 
            
        # determine if we are returning an output or setting attributes
        if output:
            return guess, stats
        else:
            self.pokemodel = guess
            self.stats_full = stats.loc['full_sample',:]
            self.stats_roll = stats.iloc[1:,:]

# %%

class pokemon_go(object):
    """
    Pokemon Go
    
    Aggregation class for Pokemon model, will run several Pokemon models at once
    and store the data internally. Option also available to generate Plotly plot.
    Name is a nod to the iPhone Game Will & I were playing during first build.
    """
    
    def __init__(self, regions=['US', 'UK', 'DR', 'JP']):
        self.regions = regions
        self.data, self.stats, self.pokemodel = [dict.fromkeys(regions)] * 3

    def download_and_run(self, plot=False, output=False):
        """
        Iterate through self.regions, download Quandl data & run Pokemon model
        Store all data in self.data, self.stats & self.pokemodel
        """
        
        for r in self.regions:        # iterate through each region
            x = pokemon(region=r)     # setup pokemon class for r
            x.download_and_run()      # download from Quandl & run
            
            # Store data in go dictionaries
            self.data[r] = x.data
            self.stats[r] = {'full':x.stats_full, 'rolling':x.stats_roll}
            self.pokemodel[r] = x.pokemodel
        
        if plot: self.plotlyplot()    # plotly dict
        
        if output:
            return self.pokemodel, self.stats, self.data
    
    def plotlyplot(self, dx=None, output=False):
        """
        Plotly Plot for Pokemon Model
        Two panes with Absolute numbers at the top & Relatives below
        Buttons to select between individual Regions/Models
        
        OUTPUT:
            Plotly dictionary of form dict(data=data, layout=layout)
            Still needs to be converted into PLotly Graph Object externally
        
        NB/ Needs to be updated when High Level Plotly thing developed
        """
                
        # Data - Setup as dictionary of Dataframes
        dx = self.pokemodel if dx == None else dx
        
        # Colourmap for charts
        cmap = {0:'purple', 1:'turquoise', 2:'grey', 3:'black', 4:'lime', 5:'blue', 6:'orange', 7:'red'}
        linewidth = [1]
        
        # Calc Abs(Max) relative from All pokemons - to use as range for yaxis2
        mxFunc = lambda r, i: np.max(abs(dx[r].iloc[:,2]-dx[r].iloc[:,i]))
        mx = np.max([np.max([mxFunc(r,0), mxFunc(r,1)]) for r in dx])
        
        # Plotly Layout Dictionary
        layout = dict(title= 'Pokemon Fair Value Bond Model',
                      font = {'family':'Courier New', 'size':12},
                      height = 500,
                      margin = {'l':100, 'r':50, 'b':50, 't':50},
                      legend = {'orientation':'h'},
                      xaxis1= {'domain': [0, 1], 'anchor': 'y1', 'title': '',},
                      xaxis2= {'domain': [0, 1], 'anchor': 'y2', 'title': '',},
                      yaxis1= {'domain': [0.42, 1.0], 'anchor': 'x1', 'title': '',
                               'hoverformat':'.2f', 'tickformat':'.1f'},
                      yaxis2= {'domain': [0.0, 0.38], 'anchor': 'x2', 'title': 'relative', 'dtick':0.5,
                               'hoverformat':'.2f', 'tickformat':'.1f', 'range':[-mx, +mx]}, 
                      updatemenus= [dict(type='buttons', active=-1, showactive = True,
                                         direction='down', pad = {'l':0, 'r':35, 't':0, 'b':0},
                                         y=1, x=0,
                                         buttons=[])],
                     annotations=[])
    
        # Plotly Data
        data = []
        for j, r in enumerate(dx.keys()):
            
            # Top plot - Absolute
            for i, n in enumerate(['full_sample', 'rolling', 'long_bond']):
                data.append(dict(type='scatter', name=n, xaxis='x1', yaxis='y1',
                                 x=dx[r].index, y=dx[r].iloc[:,i],
                                 visible=True if j==0 else False,
                                 showlegend=True,
                                 line=dict(color=cmap[i], width=linewidth[0]),))
            
            # Bottom plot - Relative to Long Bond
            for i, n in enumerate(['full_sample', 'rolling']):
                data.append(dict(type='scatter', name=n, xaxis='x2', yaxis='y2',
                                 x=dx[r].index, y=dx[r].iloc[:,2]-dx[r].iloc[:,i],
                                 visible=True if j==0 else False,
                                 fill='tozeroy', showlegend=False,
                                 line=dict(color=cmap[i], width=linewidth[0])))
            
            # Append Button
            vislist = []                             # Create a list for visible/hidden on args            
            if j==0: l = len(data)                   # Find len of each key
            for x in range(len(dx.keys())):          # build list of True or Falses for all traces
                if x == j:
                    vislist.extend([True] * l)
                else:
                    vislist.extend([False] * l)
            
            # Actual button stuffs
            button= dict(label= r, method = 'update',
                         args = [{'visible': vislist},
                                 {'title': 'Pokemon FV - {}'.format(r),}])
    
            layout['updatemenus'][0]['buttons'].append(button)    # append the button 
            
        if output:
            return dict(data=data, layout=layout)
        else:
            self.plot = dict(data=data, layout=layout)