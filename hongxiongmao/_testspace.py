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

# %%

dl = quandl_hxm()
#cli = dl.from_tickerdict('oecd_cli', start_date='-20y') - 100
#gdp = dl.from_tickerdict('oecd_normalised_gdp', start_date='-20y') - 100

# %%

# CLI Developed Markets
#cli_dm = cli.loc[:,['US', 'UK', 'EURO', 'JAPAN', 'AUS', 'CANADA', 'KOREA', 'OECD']]
#cli_groups_dm = dict(ALL=list(cli_dm.columns), Major=['US', 'EURO', 'UK', 'JAPAN'],
#                     Other=['AUS', 'CANADA', 'KOREA', 'OECD'])

#fig = overplot.swirlygram(cli.iloc[:,0:5], title='OECD Normalised CLI Swirlygram',
#                 yaxis={'title':'some shit'},)


fig = overplot.swirlygram_generalised(cli.iloc[:,0:3], gdp.iloc[:,0:3],
                                      animation=True, minax=[1, 1],
                                      duration=10, transition=2000)

py.plot(fig, auto_play=False)

