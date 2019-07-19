#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTING: Overplot Swirlygram
"""

import numpy as np
import pandas as pd
import plotly.offline as py

import overplot
from download import quandl_hxm
dl = quandl_hxm()

# %%

us_recession_watch = {
    'Recession_Probability':dict(ticker='FRED/RECPROUSM156N', name=''),
    'ISM':dict(ticker='ISM/MAN_PMI', name='ISM PMI Composite'),
    '10yr':dict(ticker='USTREASURY/YIELD', fields=['10 YR'], name='ISM PMI Composite')
    }

dl.from_tickerdict(us_recession_watch)