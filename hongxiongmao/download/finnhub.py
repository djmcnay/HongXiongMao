#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Finnhub Download Wrapper

Currently (March '20) Finnhub is my favourite freemium financial data provider; 
they appear to have a decent selection of data, have a generous free calls limit, 
the API seems fairly clean and they have been easy to contact with questions.

https://finnhub.io/

There does already exist a high level wrapper for Finnhub by s0h3ck at
https://github.com/s0h3ck/finnhub-api-python-client/ but I wanted to build into HXM.
In part to learn more about REST but also to extend functionality, eg:
    - multiple tickers
    - more logical input dates
    - output as dataframe

"""

import requests
import json
import pandas as pd
from datetime import datetime as Datetime
#from hongxiongmao import config

class call(object):
    
    def __init__(self, **kwargs):
        
        self.API_KEY = kwargs['API']
        self.URL = "https://finnhub.io/api/v1"    # base URL for requests
        
        # setup a persistent session for requests (rather than just using get())
        # session allows us to reuse API_KEYs etc...
        # https://requests.readthedocs.io/en/latest/user/advanced/#session-objects
        self.session = requests.session()
        self.session.headers.update({'Accept': 'application/json',
                                     'User-Agent': 'finnhub/python'})
        
    
    
        return
        
    def _get(self, path, **kwargs):
        
        # build url for request i.e. 
        # "https://finnhub.io/api/v1/stock/exchange?token='
        request_url = "{}/{}".format(self.URL, path)
                
        kwargs['timeout'] = 10
        data = kwargs.get('data', None)

        if data and isinstance(data, dict):
            kwargs['data'] = data
        else:
            kwargs['data'] = {}

        kwargs['data']['token'] = self.API_KEY
        kwargs['params'] = kwargs['data']

        del(kwargs['data'])

        response = getattr(self.session, 'get')(request_url, **kwargs)
        
        return response
    
    def tickers(self, **kwargs):
        return self._get("stock/symbol")
    
    def OHLC(self, **kwargs):
        return self._get()
    
        #self.client = self.Finnhub.Client(api_key=self.API_KEY)
        
# %%
        
dl = call(API='bp0pf5vrh5r9fdeibfbg')
r = dl.tickers()

r.text