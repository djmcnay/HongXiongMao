#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HongXiongMao Download Classes

Provides high level wrappers for data providers, currently:
    1. Quandl
    2. AlphaVantage
    
"""

# %%

import pandas as pd
from time import sleep
from hongxiongmao import utilities        # From HXM

# %% QUANDL

class quandl_hxm(object):
    """
    Quandl HongXiongMao Download Class
    
    """
    
    import quandl    # import here to avoid formal dependancy
    
    def __init__(self, **kwargs):
        
        # Set Quandl API Key
        if 'API' in kwargs.keys():
            self.API_KEY = kwargs['API']
        else:
            from config import API_QUANDL as API_KEY
            self.API_KEY = API_KEY
            
        self.quandl.ApiConfig.api_key = self.API_KEY

        return

    # Quandl Default Options
    quandl_opts = {'returns':'pandas',
                   'freq':'monthly',
                   'transformation':'none',
                   'start_date':'-12m',
                   'end_date':'today',
                   'date_format':'%Y-%m-%d'}
    
    # %% Useful default tickers
    
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
    
    # %% TIMESERIES DOWNLOAD
    
    def timeseries(self, tickers, fields=[], **kwargs):
        """
        Quandl Timeseries
        """
        
        # Ensure tickers is a list & update download options
        tickers = [tickers] if isinstance(tickers, str) else tickers
        
        opts = self.quandl_opts    # copy default dictionary
        for k in kwargs.keys():         
            if k in opts:
                opts[k]=kwargs[k]
                
        # Check which fields have been requested
        # Quandl by default will download all related fields
        # Use ticker[0] to establish which fields are available
        n = len(fields)
        if n > 0:
            
            # Find all available fields, then the idx of desired fields
            tick_flds = list(self.quandl.get(tickers[0], rows=1, returns='pandas').columns.values)
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
        start_date, end_date = utilities.relative_date(opts['start_date'],
                                                       opts['end_date'],
                                                       as_string = True)
        # Quandl Call
        df = self.quandl.get(fld_tickers,
                        returns = opts['returns'],
                        collapse = opts['freq'],
                        transformation = opts['transformation'],
                        start_date = start_date,
                        end_date = end_date)
        
        # If only 1 field requested then adjust column names to just be tickers
        df.columns = tickers if n == 1 else df.columns
        
        return df
    
    # %% Quandl Commodity Curves
    
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
            x = self.timeseries(tickers=tickers, fields=d['field'],
                                freq=freq, start_date=start_date)
            
            # Rename columns headers to curve tennor, then add to output
            x.columns=list(range(1,d['term']+1))
            data[f] = x.copy()
            
        return data
        
# %% ALPHAVANTAGE
        
class alphavantage(object):
    """
    AlphaVantage Download module
    
    HXM high level wrapper for RomelTorres AlphaVantage library
    """
    
    from alpha_vantage.timeseries import TimeSeries
        
    def __init__(self, **kwargs):
        
        # Set Quandl API Key
        if 'API' in kwargs.keys():
            self.API_KEY = kwargs['API']
        else:
            from config import API_ALPHAVANTAGE as API_KEY
            self.API_KEY = API_KEY
        
        # Configure API Keys for AV Modules
        self.ts = self.TimeSeries(key=self.API_KEY, output_format='pandas', retries=100000)
        return
        
    # %%
    
     # AlphaVantage Static Dictionary
    us_etfs = {
        'ACWI':dict(ticker='ACWI', name='iShares MSCI ACWI ETF'),
        'SPY':dict(ticker='SPY', name='SPDR S&P 500 ETF'),
        'QQQ':dict(ticker='QQQ', name='INVESCO QQQ TRUST QQQ USD DIS'),
        'VEA':dict(ticker='VEA', name='Vanguard FTSE Developed Markets ex US ETF'),
        'IEMG':dict(ticker='IEMG', name='iShares Core MSCI Emerging Markets ETF'),
        'VWO':dict(ticker='VWO', name='Vanguard FTSE Emerging Markets ETF'),
        'ASHR':dict(ticker='ASHR', name='Xtrackers Harvest CSI 300 China A-Shares ETF'),
        'VNQ':dict(ticker='VNQ', name='Vanguard US Real Estate Index Fund'),
        'AGG':dict(ticker='AGG', name='iShares Core U.S. Aggregate Bond ETF'),
        'GOVT':dict(ticker='GOVT', name='iShares U.S. Treasury Bond ETF'),
        'TIP':dict(ticker='TIP', name='iShares TIPS Bond ETF'),
        'LQD':dict(ticker='LQD', name='iShares iBoxx $ IG Corporate Bond ETF'),
        'HYG':dict(ticker='HYG', name='iShares iBoxx $ HY Corporate Bond ETF'),
        'EMB':dict(ticker='EMB', name='iShares J.P. Morgan USD Emerging Markets Bond ETF'),
        'EMLC':dict(ticker='EMLC', name='VanEck Vectors J.P. Morgan EM Local Currency Bond ETF'),
        'GLD':dict(ticker='GLD', name='SPDR Gold Trust'),
        'BIL':dict(ticker='BIL', name='SPDR Barclays 1-3 Month T-Bill ETF'),
        }
    
    # %% Download from AV Ticker Dictionary
    
    def dl_from_ticker_dict(self, ticker_dict, freq='d', output_size='compact'):
        """ 
        Download from Ticker Dictionary
        Uses dict of the form {'varname':{'ticker':ticker}} & downloads as df
        """
        
        # Create lists of variable names and tickers from av_tickers dict
        varnames = list(ticker_dict.keys())
        tickers = []
        for v in varnames:
            tickers.append(ticker_dict[v]['ticker'])
                    
        return self.timeseries(tickers, varnames, 
                               freq=freq, output_size=output_size)
    
    # %% AlphaVantage Timeseries
    
    def timeseries(self, tickers, varname=None, freq='d', output_size='compact'):
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
           
        # Select which download func to use based on frequency
        # RomelTorres Library has a new function for each type 
        if freq in ['d', 'daily']: api = self.ts.get_daily_adjusted
        elif freq in ['w', 'weekly']: api = self.ts.get_weekly_adjusted
        elif freq in ['m', 'monthly']: api = self.ts.get_monthly_adjusted
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
            
            # Attempting to put in a break between calls to help with latency
            print('AlphaVantage call on {} downloaded'.format(t))
            
        return data