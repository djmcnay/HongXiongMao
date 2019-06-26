# -*- coding: utf-8 -*-

# %% Dependencies

import numpy as np
import pandas as pd

# %%

def principal_drivers_index(df, n=26, px=True, min_assets=0.5):
    """
    Principal Drivers Index
    
    Calculates a correlations index by finding percentage variance explained
    by the principal eigenvector of a correlation matrix
    
    """
    
    # require a dataframe of returns
    rtn = df.pct_change() if px == True else df 
    
    # Shrink rtns input where timeseries has rtns for fewer than X% of inputs assets
    rtn = rtn.loc[rtn.notna().sum(axis=1) >= rtn.shape[1] * min_assets, :]
    
    # Dummy Dataframes - use Multi-Index for correlations
    pcr = pd.DataFrame(index=rtn.index[n:], columns=[n])
    cum = pd.DataFrame(index=rtn.index[n:], columns=range(1, rtn.shape[1]+1))
    cor = pd.DataFrame(index=pd.MultiIndex.from_product([rtn.index[n:], rtn.columns]), columns=rtn.columns)
    
    for i, v in enumerate(pcr.index):    # iterate through each date
        
        c = rtn.iloc[i:i+n,:].dropna(axis=1, how='any').corr()    # correlation matrix
        cor.loc[pd.IndexSlice[v, c.columns.tolist()], c.columns.tolist()] = c.values
        
        eig = np.linalg.eigvals(c)                                # calculate eigenvalues
        eig = eig / sum(eig)                                      # normalise
        pcr.loc[v] = eig[0]                                       # percentage of 1st eigenvalue
        cum.iloc[i, 0:len(eig)] = np.cumsum(eig)                  # cumulative sum of eigenvalues
        
    return pcr, cum, cor