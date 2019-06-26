#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:20:28 2019

@author: David
"""
import pickle
from download import alphavantage as av
import utilities

# %%

dl = av()

# %%

x = dl.dl_from_ticker_dict(dl.av_tickers, output_size='full')

# %%

filename='data_alphavantage_daily'
a = utilities.picklemerger(filename=filename, b=x, blend='left',
                           path='data', create_new=True, output=True)

# %%

infile = open('data/data_alphavantage_daily', 'rb')
a = pickle.load(infile)
infile.close()

px = utilities.daily2weekly(a['close'])