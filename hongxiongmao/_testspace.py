#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTING: Overplot Swirlygram
"""

import numpy as np
import pandas as pd
import plotly.offline as py

# %%

#import hongxiongmao as hxm

#dl = hxm.download.quandl_hxm(API='-gP4AZzsup26hKNAcsv2')

td = {
    'UST':{'ticker':'CHRIS/CME_US1', 'fields':['Open', 'Settle']},
    }


td = {
    'UST':{'ticker':'USTREASURY/YIELD', 'fields':['2 YR', '10 YR']},
    'TIPS10YR':{'ticker':'USTREASURY/REALYIELD', 'fields':['10 YR'],},
    'PMI':{'ticker':'ISM/MAN_PMI', 'fields':[], 'name':'ISM Mfg'},
    }

# %%

import download 

dl2 = download.quandl_hxm(API='-gP4AZzsup26hKNAcsv2')
px = dl2.from_tickerdict(td)

