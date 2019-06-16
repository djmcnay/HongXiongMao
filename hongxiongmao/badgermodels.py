# -*- coding: utf-8 -*-

# %% Dependencies

import numpy as np
import pandas as pd

from utilities import tools, download

# %% POKEMON

class pokemon(object):
    """
    Pokemon Fair Value Bond Model
    
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
    
    To run default we use function: pokemon_run()        
    Or to just run the regression func use: pokemon()
        - full notes available in the respective function helps
        
    Development
        * Using pre-calculated super long term coeffs rather than calculating
        * Add CSV functionality
    """
    
    # Class Dependencies
    from sklearn.linear_model import LinearRegression
    
    # Initialise
    def __init__(self, region='US'):
        self.region = region
        return
    
    # Set up Quandl download as an option
    dl = download()
    tools = tools()
    
    # %% Default Data & Attributes
    
    # Quandl
    # US - fairly happy
    quandl_us = {'US10YR':{'tickers':'USTREASURY/YIELD','fields':['10 YR']},
                 'US02YR':{'tickers':'USTREASURY/YIELD','fields':['2 YR']},
                 'UMich':{'tickers':'UMICH/SOC33','fields':['Median']},
                 'ISM':{'tickers':'ISM/MAN_PMI', 'fields':[]},
                 'REER':{'tickers':'BIS/EM_MRBUS','fields':[]}
                 }
    
    # Bunds - Need better 2yr, Inflation & Growth
    quandl_bund = {'DR10YR':{'tickers':'BUNDESBANK/BBK01_WT1010','fields':[]},
                   'DR02YR':{'tickers':'BUNDESBANK/BBK01_WT0202','fields':[]},
                   'CPI':{'tickers':'RATEINF/INFLATION_DEU','fields':[]},
                   'OECD_CLI':{'tickers':'OECD/MEI_CLI_LOLITOAA_DEU_M', 'fields':[]},
                   'REER':{'tickers':'BIS/EM_MRBXM','fields':[]}
                   }
    
    # Gilts
    quandl_gilt = {'UK10YR':{'tickers':'BOE/IUDMNZC','fields':[]},
                   'UK3MLIBOR':{'tickers':'ECB/FM_M_GB_GBP_RT_MM_GBP3MFSR__HSTA','fields':[]},
                   'CPI':{'tickers':'RATEINF/INFLATION_GBR','fields':[]},
                   'OECD_CLI':{'tickers':'OECD/MEI_CLI_LOLITONO_GBR_M', 'fields':[]},
                   'REER':{'tickers':'BIS/EM_MRBGB','fields':[]}
                   }
    
    # JGBs
    quandl_jgb = {'JP10YR':{'tickers':'MOFJ/INTEREST_RATE_JAPAN_10Y','fields':[]},
                  'JP02YR':{'tickers':'MOFJ/INTEREST_RATE_JAPAN_2Y','fields':[]},
                  'CPI':{'tickers':'RATEINF/INFLATION_JPN','fields':[]},
                  'OECD_CLI':{'tickers':'OECD/MEI_CLI_LOLITOAA_JPN_M', 'fields':[]},
                  'REER':{'tickers':'BIS/EM_MRBJP','fields':[]}
                   }
    
    quandl_dicts = {'US':quandl_us, 'DR':quandl_bund, 'UK':quandl_gilt, 'JP':quandl_jgb}
    
    @property
    def pokedata(self):
        return self.__pokedata
    @pokedata.setter
    def pokedata(self, data):              
        if isinstance(data, pd.DataFrame):
            data = data[::-1] if data.index[0] > data.index[-1] else data
        else:
            raise ValueError ('pokedata takes pd.Dataframe object; {} sent'.format(type(data)))
        self.__pokedata = data
    
    # %% Full Model Run
    
    def pokemon_run(self, region=None, start_date='10y', rolling_window=36):
        """
        Consolidation Function to Run Pokemon related stuff
        """
        region = self.region if region is None else region
        self.pull_data_from_quandl(region=region, start_date=start_date)
        self.pokemon(window=rolling_window)
        
        return
    
    # %% Import Data
        
    def pull_data_from_quandl(self, region='US', 
                              start_date='25y', freq='monthly',
                              output=False):
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
            x = self.dl.quandl_ts(tickers = j['tickers'], fields = j['fields'],
                                    start_date=start_date, freq=freq)
            
            x.columns = [v[0]]    # Update column header with dictionary key
            
            # Merge new Dataframe output df
            if i == 0:
                data = x
            else:
                data = tools.df_merger(data, x)
        
        # output determines if we return data or set attribute
        if output:
            return data
        else:
            self.pokedata = data

    # %% Pokemon Model
    
    def pokemon(self, df=None, window=36, valiadate_data=True, output=False):
        """
        Pokemon Fair Value Model
        """
        
        # Use internal data attribute if None is provided
        df = self.pokedata if df is None else df
        
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
        lm = self.LinearRegression().fit(X,y)
        guess['full_sample'] = lm.predict(X)
        stats.loc['full_sample',['r2','intercept']]=lm.score(X, y),lm.intercept_
        stats.loc['full_sample'].iloc[2:] = lm.coef_
        
        # Rolling Window
        for i, v in enumerate(guess.index):
            
            if i < window:
                continue
            
            y = df.iloc[i-window:i,0]     # Dependant Var [long bond]
            X = df.iloc[i-window:i,1:]    # Independant [short, inf, growth]
            roll_lm = self.LinearRegression().fit(X,y)
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
    
    pass    # END of Pokemon model
    
# %% TEST
    
us = pokemon(region='US')
us.pokemon_run()
us.pokemon

# %%

us.pokemodel
    
    