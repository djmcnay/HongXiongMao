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

dl = quandl_hxm()
cli = dl.from_tickerdict('oecd_cli', start_date='-20y') - 100

# %%

# CLI Developed Markets
cli_dm = cli.loc[:,['US', 'UK', 'EURO', 'JAPAN', 'AUS', 'CANADA', 'KOREA', 'OECD']]
cli_groups_dm = dict(ALL=list(cli_dm.columns), Major=['US', 'EURO', 'UK', 'JAPAN'],
                     Other=['AUS', 'CANADA', 'KOREA', 'OECD'])

fig = overplot.scatter_from_dataframe(cli_dm, groups=cli_groups_dm,
                                      title='OECD Normalised CLI - Developed Markets',
                                      ytitle='Normalised CLI (minus 100)',
                                      button_position=[1.1, 0.25])

py.plot(fig)
# %%

cli_em = cli.loc[:,['CHINA', 'MEXICO', 'RUSSIA', 'INDIA', 'BRAZIL', 'TURKEY', 'ZAR', 'ASIA5']]
cli_groups_em = dict(ALL=list(cli_em.columns), BRIC=['CHINA', 'BRAZIL', 'RUSSIA', 'INDIA'],
                     Asia=['CHINA', 'INDIA', 'ASIA4'], Latam=['MEXICO', 'BRAZIL'], CEEMEA=['RUSSIA', 'TURKEY', 'ZAR'])#

fig = overplot.scatter_from_dataframe(cli_em, groups=cli_groups_em,
                                      title='OECD Normalised CLI - Emerging Markets',
                                      ytitle='Normalised CLI (minus 100)',
                                      button_position=[0, -0.1],
                                      button_direction = 'right')
py.plot(fig)
#py.plot(fig, filename='PlotlyHTMLexJS/cli_em.html', auto_open=False, include_plotlyjs='cdn',include_mathjax='cdn')

# %%


#fig = overplot.scatter_from_dataframe(cli_dm, groups=cli_groups_dm,
#                                      title='OECD Normalised CLI - Developed Markets',
#                                      ytitle='Normalised CLI (minus 100)',
#                                      button_position = [0,-0.1,0,0,0,0],
#                                      button_direction = 'right')

#py.plot(fig)