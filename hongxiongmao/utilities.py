#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HongXiongMao utility functions

"""

#import hongxiongmao.config as config        # API Keys

# Dependencies
import os
import pickle
import numpy as np
import pandas as pd
import datetime as dt

# %% DATAFRAME MERGER

# Dataframe Merger
def df_merger(a, b, blend='left'):
    """
    Merge two Pandas timeseries dataframes with inconsistent data for e.g.
    different time periods or different asset classes and return single
    dataframe with unique columns and all periods from both input dfs.
    
    Very useful for updating timeseries data where we want to keep the
    original data, but add new data. 
    
    blend can be 'left'(default), 'right' or 'mean'
        left - preserves data from a & appends from b; quick update
        right - preserves data from b & appends from a; for point in time
        mean - takes mean of (non NaN) data where there is a difference
    """
    # Concat along rows to keep all indices and columns from a & b
    # Groupby column names and apply a homemade sorting function
    # Groupby will keep duplicated columns with the same column name
    # Remove these duplicated columns
    c = pd.concat([a, b], axis=1, sort=True)
    df = c.groupby(c.columns, axis=1, sort=False).apply(
            lambda x: _df_merger_helper_func(x, blend))
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def _df_merger_helper_func(x, blend='mean'):
    """ Function is part of df_merger in Pandas, i.e. updating a timeseries 
    We pass a df x, as a result of a pd.DataFrame.groupby()
    For example x[:,0] is SPX from 1999-2017 & x[:,1] is 2016-2019
    Here we output a df with a single column that blends the dataseries
    """
    if x.shape[1] == 1:
        # 1st test shape, width==1 => series therefore return itself
        return x
    elif (x.shape[1] == 2) and (blend in ['left', 'right']):        
        # We chose which is the primary (called left) & which is the updater
        # Subset the "left" column of data, then find the index of NaNs
        # Replace nans on left with values from the "right" (could be NaN)
        # Always same length because concat func creates NaNs on missing data
        v = [0,1] if blend == 'left' else [1,0]
        l = x.iloc[:,v[0]]
        i = np.isnan(l)
        l[i] = x[i].iloc[:,v[1]]
        return pd.DataFrame(l, columns=[x.columns.values[0]])
    else:
        # Default otherwise is to return the mean
        # For mean we take the mean of x (which outputs a series)
        # Then convert back to dataframe & return the original column header
        return pd.DataFrame(x.mean(axis=1), columns=[x.columns.values[0]])

# %% DATE FUNCTIONS
        
def relative_date(r='12m', end_date='today',
                  date_format='%Y-%m-%d', as_string=False):
    """
    Relative Date function
    
    Calculates a datetime from a given end date and a relative reference.
    
    INPUT:
        r - relative date reference as '-12d' accepts d, w, m or y
        end_date - 'today' (default), date string, datetime object
        date_format - input format of string & output if requested
        as_string - True | False (default) 
                    decides if output is converted to string from datetime   
    """
    
    # Create Datetime object end_date based on supplied end_date
    # If not string or 'today' assume already in datetime format
    if end_date == 'today':
        end_date = dt.datetime.today()        
    elif isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, date_format)
        
    # Breakdown Relative Reference into type (i.e. d, w, m, y) & number
    r = r[1::] if r[0] == '-' else r
    dtype, dnum = str(r[-1]).lower(), float(r[0:-1])
    
    # Manipulate based on relative Days, Weeks, Months or Years
    if   dtype == 'd': start_date = end_date - dt.timedelta(days=dnum)
    elif dtype == 'w': start_date = end_date - dt.timedelta(weeks=dnum)
    elif dtype == 'm': start_date = end_date - dt.timedelta(weeks=dnum*4)
    elif dtype == 'y': start_date = end_date - dt.timedelta(weeks=dnum*52.143)
        
    # Output as Strings if desirable
    if as_string is True:
        start_date = dt.datetime.strftime(start_date, date_format)
        end_date = dt.datetime.strftime(end_date, date_format)

    return start_date, end_date

# Daily-to-Weekly
def daily2weekly(ts, day=4, date_str_format='%Y-%m-%d'):
    """
    daily2weekly
    
    Converts timeseries Dataframe from daily to weekly data.
    Defaulting to Friday(day=4) but others days possible (Monday=0)
    """
    dt = pd.to_datetime(ts.index, format=date_str_format).to_series().dt.dayofweek
    idx = dt.values == day    # Reindex using desired day of week
    new = ts.iloc[idx] if isinstance(ts, pd.Series) else ts.iloc[idx,:]
    return new

# %% FILE SAVING & UNLOADING

# Pickle Merger
def picklemerger(filename, b, blend='left', path=None,
                 create_new=False, output=False):
    """
    Merges a pickle file with another dictionary where the new dictionary
    is of the form {'variable':pd.DateFrame}
     * Will add keys from dict b if those keys aren't in a already.
     * can create_new file or replace original completely if required
     * can select blend method 'left' or 'right' for common non-NaN index

    INPUTS:
        b - dictionary being appended
        blend - 'left'(default)|'right'
                decides if a or b is master where there is common index
        create_new - False (default) | True | 'replace'
                     True builds new file (and dir) from b if non exists
                     replace will replace current file with b
        output - False (default)|True if we want function to return pickle
                     
    """
    
    # Make a full filepath from the path & filename
    if path == None:
        filepath = filename.lower()
    else:
        path = path+'/' if path[-1] != '/' else path
        filepath = (path+filename).lower()
            
    # Unpickle packed file
    # If replacing don't open, create new dir & set a = b
    if create_new == 'replace':
        os.makedirs(os.path.dirname(path.lower()), exist_ok=True)
        a = b
    else:
        try:
            infile = open(filepath, 'rb')
            a = pickle.load(infile)
            infile.close()
        except:
            if create_new:
                os.makedirs(os.path.dirname(path.lower()), exist_ok=True)
                a = b
            else:
                raise ValueError('ERR: no file {} at path ./{}'.format(filename, path))
    
    # Iterate through each key in original file
    for k in list(a.keys()):
        if k in list(b.keys()):
            a[k] = df_merger(a[k], b[k], blend=blend)
            
    # Add new dictionaries keys from b that aren't in a
    for k in list(b.keys()):
        if k not in list(b.keys()):
            a[k] = b[k]
    
    # Re-pickle & save
    picklefile = open(filepath, 'wb')
    pickle.dump(a, picklefile)
    picklefile.close()
    
    print('{} updated & Pickled in {}'.format(filename, path))
    
    if output: return a
    else: return