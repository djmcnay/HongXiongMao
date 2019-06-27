#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:20:28 2019

@author: David
"""
import pickle
from download import alphavantage as av
import utilities
#import badgermodels as bm
from principaldrivers import pdi

# %%

#dl = av()
#x = dl.dl_from_ticker_dict(dl.us_etfs, output_size='full')
#filename='data_alphavantage_daily'
#a = utilities.picklemerger(filename=filename, b=x, blend='left',
#                           path='data', create_new=True, output=True)

# %%

infile = open('data/data_alphavantage_daily', 'rb')
a = pickle.load(infile)
infile.close()

etfs=['SPY', 'VEA', 'VWO', 'GOVT', 'TIP', 'LQD', 'HYG', 'EMB', 'EMLC', 'GLD']
px = utilities.daily2weekly(a['close']).loc[:,etfs]
#pcr = bm.principal_drivers_index(px)


# %%

p = pdi(ts=px)
p.principal_drivers_index()
p.plotly_pdi()
print(p.plotlyplot)