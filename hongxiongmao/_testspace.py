#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:20:28 2019

@author: David
"""

from download import alphavantage as av

dl = av()

# %%

x = dl.dl_from_ticker_dict(dl.av_tickers, output_size='full')