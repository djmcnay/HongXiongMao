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

#dl = quandl_hxm()
#cli = dl.from_tickerdict('oecd_cli', start_date='-5y')

# %%

#cli = cli - 100
#cli_dm = cli.loc[:,['US', 'UK', 'EURO', 'JAPAN', 'AUS', 'CANADA', 'KOREA', 'OECD']]
#cli_groups_dm = dict(ALL=list(cli_dm.columns), Major=['US', 'EURO', 'UK', 'JAPAN'], Other=['AUS', 'CANADA', 'KOREA', 'OECD'])

# %%

fig = overplot.scatter_from_dataframe(cli_dm, groups=cli_groups_dm,
                                      title='OECD Normalised CLI - Developed Markets',
                                      ytitle='Normalised CLI (minus 100)',
                                      button_position = [0,0,0,0,0,0],
                                      button_direction = 'down')

py.plot(fig)