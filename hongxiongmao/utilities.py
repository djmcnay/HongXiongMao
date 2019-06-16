#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
badgertools utility functions


"""

import hongxiongmao.config as config        # API Keys

# Dependencies
import os
import pickle
import numpy as np
import pandas as pd
from time import sleep
import datetime as dt

# %%

class tools(object):
    """
    
    """
    
    def __init__(self):
        return
    
    # Pickle Merger
    def picklemerger(self, filename, b, blend='left', path=None,
                     create_new=False, output=False):
        """
        Merges a pickle file with another dictionary where the new dictionary
        is of the form {'variable':pd.DateFrame}
         * Will add keys from dict b if those keys aren't in a already.
         * can create_new file or replace original completely if required
         * can select blend method 'left' or 'right' for common non-NaN index

        INPUTS:
            b - dictionary being appended
            blend - 'left'(default)|'right'
                    decides if a or b is master where there is common index
            create_new - False (default) | True | 'replace'
                         True builds new file (and dir) from b if non exists
                         replace will replace current file with b
            output - False (default)|True if we want function to return pickle
                         
        """
        
        # Make a full filepath from the path & filename
        if path == None:
            filepath = filename.lower()
        else:
            path = path+'/' if path[-1] != '/' else path
            filepath = (path+filename).lower()
                
        # Unpickle packed file
        # If replacing don't open, create new dir & set a = b
        if create_new == 'replace':
            os.makedirs(os.path.dirname(path.lower()), exist_ok=True)
            a = b
        else:
            try:
                infile = open(filepath, 'rb')
                a = pickle.load(infile)
                infile.close()
            except:
                if create_new:
                    os.makedirs(os.path.dirname(path.lower()), exist_ok=True)
                    a = b
                else:
                    raise ValueError('ERR: no file {} at path ./{}'.format(
                                     filename, path))
        
        # Iterate through each key in original file
        for k in list(a.keys()):
            if k in list(b.keys()):
                a[k] = self.df_merger(a[k], b[k], blend=blend)
                
        # Add new dictionaries keys from b that aren't in a
        for k in list(b.keys()):
            if k not in list(b.keys()):
                a[k] = b[k]
        
        # Re-pickle & save
        picklefile = open(filepath, 'wb')
        pickle.dump(a, picklefile)
        picklefile.close()
        
        if output:
            return a
        else:
            return
    
    # Dataframe Merger
    def df_merger(self, a, b, blend='left'):
        """
        Merge two Pandas timeseries dataframes with inconsistent data for e.g.
        different time periods or different asset classes and return single
        dataframe with unique columns and all periods from both input dfs.
        
        Very useful for updating timeseries data where we want to keep the
        original data, but add new data. 
        
        blend can be 'left'(default), 'right' or 'mean'
            left - preserves data from a & appends from b; quick update
            right - preserves data from b & appends from a; for point in time
            mean - takes mean of (non NaN) data where there is a difference
        """
        # Concat along rows to keep all indices and columns from a & b
        # Groupby column names and apply a homemade sorting function
        # Groupby will keep duplicated columns with the same column name
        # Remove these duplicated columns
        c = pd.concat([a, b], axis=1, sort=True)
        df = c.groupby(c.columns, axis=1, sort=False).apply(
                lambda x: self._df_merger_helper_func(x, blend))
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def _df_merger_helper_func(self, x, blend='mean'):
        """ Function is part of df_merger in Pandas, i.e. updating a timeseries 
        We pass a df x, as a result of a pd.DataFrame.groupby()
        For example x[:,0] is SPX from 1999-2017 & x[:,1] is 2016-2019
        Here we output a df with a single column that blends the dataseries
        """
        if x.shape[1] == 1:
            # 1st test shape, width==1 => series therefore return itself
            return x
        elif (x.shape[1] == 2) and (blend in ['left', 'right']):        
            # We chose which is the primary (called left) & which is the updater
            # Subset the "left" column of data, then find the index of NaNs
            # Replace nans on left with values from the "right" (could be NaN)
            # Always same length because concat func creates NaNs on missing data
            v = [0,1] if blend == 'left' else [1,0]
            l = x.iloc[:,v[0]]
            i = np.isnan(l)
            l[i] = x[i].iloc[:,v[1]]
            return pd.DataFrame(l, columns=[x.columns.values[0]])
        else:
            # Default otherwise is to return the mean
            # For mean we take the mean of x (which outputs a series)
            # Then convert back to dataframe & return the original column header
            return pd.DataFrame(x.mean(axis=1), columns=[x.columns.values[0]])
        
    def relative_date(self, r='12m', end_date='today',
                      date_format='%Y-%m-%d', as_string=False):
        """
        Relative Date function
        
        Calculates a datetime from a given end date and a relative reference.
        
        INPUT:
            r - relative date reference as '-12d' accepts d, w, m or y
            end_date - 'today' (default), date string, datetime object
            date_format - input format of string & output if requested
            as_string - True | False (default) 
                        decides if output is converted to string from datetime   
        """
        
        # Create Datetime object end_date based on supplied end_date
        # If not string or 'today' assume already in datetime format
        if end_date == 'today':
            end_date = dt.datetime.today()        
        elif isinstance(end_date, str):
            end_date = dt.datetime.strptime(end_date, date_format)
            
        # Breakdown Relative Reference into type (i.e. d, w, m, y) & number
        r = r[1::] if r[0] == '-' else r
        dtype, dnum = str(r[-1]).lower(), float(r[0:-1])
        
        # Manipulate based on relative Days, Weeks, Months or Years
        if   dtype == 'd': start_date = end_date - dt.timedelta(days=dnum)
        elif dtype == 'w': start_date = end_date - dt.timedelta(weeks=dnum)
        elif dtype == 'm': start_date = end_date - dt.timedelta(weeks=dnum*4)
        elif dtype == 'y': start_date = end_date - dt.timedelta(weeks=dnum*52.143)
            
        # Output as Strings if desirable
        if as_string is True:
            start_date = dt.datetime.strftime(start_date, date_format)
            end_date = dt.datetime.strftime(end_date, date_format)
    
        return start_date, end_date
    
# %%
    
class download(object):
    """
    Download Module for BadgerTools
    
    Offers specific usecase download & pickling for
        * AlphaVantage
        * Quandl
    
    """
    
    def __init__(self, **kwargs):
        
        # Update API Keys from config file 
        self.api_quandl = config.api_quandl
        self.api_av = config.api_alphavantage
        
        # Default data drive if non-provided as kwarg
        self.datafolder = kwargs['path'] if 'path' in kwargs else 'data/'
        
        # toolkit from within Utils
        self.tools = tools()
        
        return
    
    # %% Class Attributes
    
    @property
    def datafolder(self):
        return self.__datafolder
    @datafolder.setter
    def datafolder(self, data):              
        if isinstance(data, str): self.__cot_data = data
        else: raise ValueError ('Err: Invalid datafolder path {}'.format(type(data)))
        
    # %% Global Static
    
    # AlphaVantage Static Dictionary
    av_tickers = {
        'MXWD':dict(ticker='ACWI', name='iShares MSCI ACWI ETF'),
        'SPX':dict(ticker='SPY', name='SPDR S&P 500 ETF'),
        'EAFE':dict(ticker='VEA', name='Vanguard FTSE Developed Markets ex US ETF'),
        'MXEF':dict(ticker='IEMG', name='iShares Core MSCI Emerging Markets ETF'),
        'CSI300':dict(ticker='ASHR', name='Xtrackers Harvest CSI 300 China A-Shares ETF'),
        'REITS':dict(ticker='IYR', name='iShares U.S. Real Estate ETF'),
        'US_Agg':dict(ticker='AGG', name='iShares Core U.S. Aggregate Bond ETF'),
        'US_Govt':dict(ticker='GOVT', name='iShares U.S. Treasury Bond ETF'),
        'TIPS':dict(ticker='TIP', name='iShares TIPS Bond ETF'),
        'US_IG':dict(ticker='LQD', name='iShares iBoxx $ IG Corporate Bond ETF'),
        'US_HY':dict(ticker='HYG', name='iShares iBoxx $ HY Corporate Bond ETF'),
        'EMD_Hard':dict(ticker='EMB', name='iShares J.P. Morgan USD Emerging Markets Bond ETF'),
        'EMD_Local':dict(ticker='EMLC', name='VanEck Vectors J.P. Morgan EM Local Currency Bond ETF'),
        'GOLD':dict(ticker='GLD', name='SPDR Gold Trust'),
        'USD':dict(ticker='SHV', name='iShares Short Treasury Bond ETF'),
        }
    
    # Quandl Commodity Futures Curves
    quandl_cmdty_curves = {
        'Brent':dict(ticker='CHRIS/ICE_B', term=24, field='Settle', name='ICE Brent Crude'),
        'Nat Gas':dict(ticker='CHRIS/CME_NG', term=24, field='Settle', name='CME Natural Gas'),
        'WTI':dict(ticker='CHRIS/ICE_T', term=7, field='Settle', name='ICE WTI Crude'),
        'XAU':dict(ticker='CHRIS/CME_GC', term=10, field='Settle', name='Gold'),
        'XAG':dict(ticker='CHRIS/CME_SI', term=9, field='Settle', name='Silver'),
        'XPT':dict(ticker='CHRIS/CME_PL', term=2, field='Settle', name='Platinum'),
        'Copper':dict(ticker='CHRIS/CME_HG', term=15, field='Settle', name='Copper'),
        'VIX':dict(ticker='CHRIS/CBOE_VX', term=9, field='Settle', name='Vix'),
        }

    # Quandl Default Options
    quandl_opts = {'returns':'pandas',
                   'freq':'monthly',
                   'transformation':'none',
                   'start_date':'-12m',
                   'end_date':'today',
                   'date_format':'%Y-%m-%d'}
    
    # %% Default Downloads
    
    # AlphaVantage default OHLC timeseries
    def update_alphavantage_ts(self, output_size='compact', output=True,
                               filename=None, path=None):
        """
        Calls full default ticker list from AlphaVantage and stores OHLC Data.
        Then updates pickle file from data/ directory using left-blend
        """
        
        # Create lists of variable names and tickers from av_tickers dict
        varnames = list(self.av_tickers.keys())
        tickers = []
        for v in varnames:
            tickers.append(self.av_tickers[v]['ticker'])
            
        # Call data from AV
        b = self.alphavantage_ts(tickers, varnames, freq='d', output_size=output_size)
        
        # Update data store
        filename='data_alphavantage_daily'
        a = self.tools.picklemerger(filename=filename, b=b, blend='left',
                           path='data', create_new=False, output=True)
        
        if output: return a
        else: return
        
    def cmdty_futures_curves(self, tickdic=None, start_date='-5y', freq='w',
                             output=True, store=True, ):
        """
        Close PX of Commodity Futures across multiple tennors
        Data pulled from Quandl Wiki Futures; my default contracts supplied
        Stored in dict of form {'contract':pd.DataFrame(columns=tennors)}
        
        INPUT:
            tickdic - dictionary of the form {'Contract Name':{sub-dictionary}}
                where sub-dict MUST have keys ['ticker', 'fields', 'term']
                e.g. {'Brent':{'ticker':'CHRIS/ICE_B','term':24,'field':'Settle'}
    
        """
        
        # Use class default input dictionary if None provided
        tickdic = self.quandl_cmdty_curves if tickdic is None else tickdic
        
        # Iterate through each contract in input dictionary
        data = dict()    # Dummy output dictionary
        for f in list(tickdic.keys()):
            
            # Sub dict with keys [tickers, field, name, term]
            # Quandl wiki futures have a base ticker+term ICE_BX, where X is term  
            # create ticker list of form ['ICE_B1', 'ICE_B23'] for all tennors
            d = tickdic[f]
            tickers = [d['ticker']+str(i) for i in list(range(1,d['term']+1))]  
            
            # Call Quandl to pull timeseries data
            x = self.quandl_ts(tickers=tickers, fields=d['field'],
                             freq=freq, start_date=start_date)
            
            # Rename columns headers to curve tennor, then add to output
            x.columns=list(range(1,d['term']+1))
            data[f] = x.copy()
            
        return data
        
    # %% Quandl
    
    def quandl_ts(self, tickers, fields=[], **kwargs):
        """
        Quandl Timeseries
        """
        
        # Initialisation basics
        import quandl
        quandl.ApiConfig.api_key = self.api_quandl
        tickers = [tickers] if isinstance(tickers, str) else tickers
        
        # Get Quandl Defaults and update on adhoc basis
        opts = self.quandl_opts
        for k in kwargs.keys():         
            if k in opts:
                opts[k]=kwargs[k]
                
        # Check which fields have been requested
        # Quandl by default will download all related fields
        # Use ticker[0] to establish which fields are available
        n = len(fields)
        if n > 0:
            
            # Find all available fields, then the idx of desired fields
            tick_flds = list(quandl.get(tickers[0], rows=1, returns='pandas').columns.values)
            fld_idx = [str(i+1) for i, x in enumerate(tick_flds) if x in fields]   
            
            if len(fld_idx) == 0:
                raise ValueError('fld_idx empty; fields for ticker {} are {}'
                                 .format(tickers[0], tick_flds))
            
            # Update Ticker list with numerical index of desired fields
            fld_tickers=list()
            zippy = list(zip(tickers * n, fld_idx * len(tickers)))
            [fld_tickers.append('.'.join(j)) for j in zippy]
        else:
            fld_tickers=tickers
            
        # Work out start & end dates and convert to Quandl string format
        start_date, end_date = self.tools.relative_date(opts['start_date'],
                                                        opts['end_date'],
                                                        as_string = True)
        # Quandl Call
        df = quandl.get(fld_tickers,
                        returns = opts['returns'],
                        collapse = opts['freq'],
                        transformation = opts['transformation'],
                        start_date = start_date,
                        end_date = end_date)
        
        # If only 1 field requested then adjust column names to just be tickers
        df.columns = tickers if n == 1 else df.columns
        
        return df
    
    # %% AlphaVantage
        
    def alphavantage_ts(self, tickers, varname=None, freq='d', output_size='compact'):
        """
        Downloads stock closing price data from AlphaVantage, returning pd dataframe
         
        INPUTS:
            tickers - 'string' or [list] of tickers
            varname - None(default) or list of names to pass for column headers
            freq - 'daily'(default)|'weekly'|'monthly'
            output_size - 'compact'(default)|'full' & defined by RT AV Lib
                1. 'compact' (default) is 100 entries
                2. 'full' which is everything
        
        OUTPUT:
            dictionary of pd.Dataframes
            keys are 'open', 'high', 'low', 'close'
        
        Future Development:
            - Accept dictionaries to allow
            - Error Handling for failed tickers; currently craps out
        """
        
        # wrap tickers into list if passed as string
        # Setup varnames if None passed
        tickers = [tickers] if isinstance(tickers, str) else tickers
        varname = tickers if varname == None else varname
        
        # Instantiate TimeSeries - importing RomelTorres library
        from alpha_vantage.timeseries import TimeSeries
        ts = TimeSeries(key=self.api_quandl, output_format='pandas', retries=1000)
        
        # Select which download func to use based on frequency
        # RomelTorres Library has a new function for each type 
        if freq in ['d', 'daily']: api = ts.get_daily_adjusted
        elif freq in ['w', 'weekly']: api = ts.get_weekly_adjusted
        elif freq in ['m', 'monthly']: api = ts.get_monthly_adjusted
        else: raise ValueError('Invalid input freq - {}'.format(freq))
        
        # Iterate through each ticker in ticker list
        for i, t in enumerate(tickers):

            # Seconds gaps in case of latency problems on multiple av calls
            # mostly remediated by the retries thing but does cause some issues
            for gap in [5, 10, 15, 20, 30, 45]:
                try:
                    dl, _ = api(t, outputsize=output_size)    # AlphaVantage Call
                    break
                except:
                    print('ERR: AlphaVantage call on {}; retry in {}s'.format(t, gap))
                    sleep(gap)
                    continue
                    
            # setup output dict on 1st ticker; concat to data otherwise
            if i == 0:
                data = dict()    # dummy output dict
                data['open'] = dl.loc[:,'1. open'].rename(varname[i])
                data['high'] = dl.loc[:,'2. high'].rename(varname[i])
                data['low'] = dl.loc[:,'3. low'].rename(varname[i])
                data['close'] = dl.loc[:,'4. close'].rename(varname[i])
            else:
                data['open'] = pd.concat([data['open'],dl.loc[:,'1. open'].rename(varname[i])], axis=1, sort=True)
                data['high'] = pd.concat([data['high'], dl.loc[:,'2. high'].rename(varname[i])], axis=1, sort=True)
                data['low'] = pd.concat([data['low'], dl.loc[:,'3. low'].rename(varname[i])], axis=1, sort=True)
                data['close'] = pd.concat([data['close'], dl.loc[:,'4. close'].rename(varname[i])], axis=1, sort=True)
                
        return data