# -*- coding: utf-8 -*-
"""
Dividend Discount Model(s)

"""


# %%

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

# %% Equity Sustainable Returns Model

def multi_stage_irr(dv,
                    trend_start=10,
                    terminal=21, **kwargs):
    """ Multi-Stage Dividend Discount Model solving for IRR
    
    Assumtion is 3 phases:
        1. Convergence to trend
        2. Time in trend
        3. Perpetual growth
    
    INPUTS:
        dv = data-vector which can either be a list or a pd.Series; must contain: 
             ['PX', 'D0', 'ROE', 'PO', 'ROE_trend', 'PO_trend', 'G']
        trend_start = int of year (default=10) entering trend growth
        terminal = int or year (default=21) entering perpetual growth phase

    """
    
    if isinstance(dv, list):
        vn = ['PX', 'D0', 'ROE', 'PO', 'ROE_trend', 'PO_trend', 'G']
        dv = pd.Series(data=dv, index=vn)#.T    # Transpose as lists form column df
        
    # Set up output Dataframe, with the Index representing years
    ddm = pd.DataFrame(index=range(terminal+1), columns=['D','ROE','PO','g','PX'])
    
    # T0
    ddm.loc[0,['D', 'ROE', 'PO', 'PX']] = dv.D0, dv.ROE, dv.PO, dv.PX
        
    # Phase 1 - converge to trend    
    ddm.loc[0:trend_start, 'ROE'] = np.linspace(ddm.loc[0,'ROE'], dv.ROE_trend, trend_start+1).ravel()
    ddm.loc[0:trend_start, 'PO'] = np.linspace(ddm.loc[0,'PO'], dv.PO_trend, trend_start+1).ravel()
    # Phase 2 - time in trend
    ddm.loc[trend_start:, 'ROE'] = dv.ROE_trend
    ddm.loc[trend_start:, 'PO'] = dv.PO_trend
    
    # implied g, Terminal G & update dividends
    ddm['g'] = ddm.ROE * (1 - ddm.PO) 
    for i, v in enumerate(ddm.index[:-1]):
        ddm.loc[i+1, 'D'] = ddm.loc[i, 'D'] * (1 + ddm.loc[i, 'g'])
    
    ddm.loc[ddm.index[-1], 'g'] = dv['G']    # set Terminal Growth Rate in table
    
    # Scipy optimisation
    # solver function is _opt_irr
    # Required args are Vector Dividends, current PX and Terminal G
    res = minimize_scalar(_opt_irr, args=(ddm.D, ddm.PX[0], ddm.g.iloc[-1]))
    
    return ddm, res
    
# Optimisation Function for IRR
def _opt_irr(x, d, px, G):
    """ IRR function for Multi-Stage Dividend Discount Model in Scipy
    
    Uses the scipy.optimise minimize_scalar optimisation function
        minimize_scalar(f, args=(d, px, G)).x
        
    Where d - vector of dividend stream from d0 to dTerminal
          px - current index level associated with dividend stream
          G = perpetual growth rate """
    
    pv1 = 0        # Present Value of "phases"
    n = len(d)-1   # Year of Terminal Growth (-1 removes t0 from height)
    
    # Geometric sum of PV from year 1 to terminal
    d1 = d[1:-1].values
    for i, v in enumerate(d1):
        pv1 += d1[i] / ((1 + x) ** i) 
    
    # Calculate PV of Terminal Value
    pvT = (d[n] / ((1 + x) ** n)) * (1 / (x - G))
    
    return np.abs(pv1 + pvT - px)    # minimise in optimiser

# %%
dv = [11.9, 0.237, 0.13, 0.425, 0.14, 0.35, 0.01]
x, y = multi_stage_irr(dv)

