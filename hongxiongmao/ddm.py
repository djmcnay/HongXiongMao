# -*- coding: utf-8 -*-
"""
Dividend Discount Model(s)

Series of functions for solving dividend discount model problems. Primarily

- dividend_discount_model_irr() is a sovler for given px, divident stream & G
- multi_stage_irr() is a 3-stage model which estimates g = ROE * (1-PO)

"""

# %% REQUIRED PACKAGES

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

# %% DDM Solver Functions - given dividend stream
# Build a dividend stream using another function - solve it here

def dividend_discount_model_irr(px, vDividend, G):
    """ IRR finder for Dividend Discount Model in SciPy
    
    Uses the scipy.optimise minimize_scalar optimisation function
        minimize_scalar(f, args=(d, px, G)).x
        
    Where d - vector of dividend stream from d0 to dTerminal
          px - current index level associated with dividend stream
          G = perpetual growth rate
    """
    
    return minimize_scalar(_solver_ddm_irr, args=(px, vDividend, G)).x

def _solver_ddm_irr(x, px, d, G):
    """ solver function for dividend_discount_model_irr"""
    pv = 0     # Set present value of div stream to 0 to t0
    for i, v in enumerate(d):
        
        if i == 0:              # skip d0 which isn't included in DDM calc
            continue
        elif i < len(d) - 1:    # len(d)-1 == terminal year
            pv += v / ((1 + x) ** i)
        else:
            pv += (v / ((1 + x) ** i)) * ( 1 / (x - G))
            
    return np.abs(px - pv)    # minimise in optimiser

# %% Equity Sustainable Returns Model

def multi_stage_irr(dv,
                    trend_start=10,
                    terminal=21, **kwargs):
    """ Multi-Stage Dividend Discount Model solving for IRR
    
    Assumtion is 3 phases:
        1. Convergence to trend
        2. Time in trend
        3. Perpetual growth

    growth with growth estimated as g = ROE * (1 - Payout_Ratio)
    
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
    
    # solve for IRR using homemade function
    r = dividend_discount_model_irr(dv.PX, ddm.D, dv.G)
    
    # Show present value in table    
    for i, v in enumerate(ddm.index):
        if i == 0:
            continue
        elif v < terminal:
            ddm.loc[v,'PX'] = -1 * ddm.loc[v,'D']/((1+r)**i)
        else:
            ddm.loc[v,'PX'] = (ddm.loc[v,'D'] /((1+r)**i)) * -1 * (1 / (r - ddm.g.iloc[-1]))
            
    return ddm, r

# %%