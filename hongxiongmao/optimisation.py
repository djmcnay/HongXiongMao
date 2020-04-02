#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HongXiongMao Optimisation Functions

"""

import numpy as np
import pandas as pd
import cvxpy as cvx

import matplotlib.pyplot as plt

# %% MEAN VARIANCE OPTIMISER

class mvo(object):
    """
    Mean Variance Optimisation Class
    
    May be used as a freestanding optimiser but designed to act as subclass.
    Other convex optimisation packages available but this uses CVXPY (& CVXOPT)
    
    INPUTS:
        mu - vector of mean returns; preferably pd.DataFrame but np.array accepted
        vcv - variance-covariance matrix as np.matrix
        asset_constraints - asset class level constraints as dict;
            input of form {'name':([universe], operator, value)}
             
    """
    
    def __init__(self, mu=None, vcv=None, asset_constraints=None):
        self.gamma = cvx.Parameter(nonneg=True)       # risk aversion parameter
        self.constraints = []                         # constraints for CVXPY
        self.asset_constraints = asset_constraints    # asset class constraints list
        if mu is not None: self.mu = mu
        if vcv is not None: self.vcv = vcv        

    # Expected Returns
    # Test to Ensure vector, if Pandas then save Index as universe attribute
    # Set up weight vector w as Variable in CVXPY
    @property
    def mu(self): return self.__mu
    @mu.getter
    def mu(self): return self.__mu.value
    @mu.setter
    def mu(self, x):
        if len(np.shape(x)) != 1: raise ValueError('mvo mu input {} is non-vector'.format(x))
        else: self.w = cvx.Variable(len(x))    # weight Variable for CVXPY     
        
        if isinstance(x, pd.Series):
            self.universe = list(x.index)
            self.__mu = cvx.Parameter(len(x),value=x.values)
        else:
            self.__mu = cvx.Parameter(len(x), value=x)
        
    # Variance-Covariance Matrix
    # Checks vcv is symmetrical & no -ve variances
    @property
    def vcv(self): return self.__vcv
    @vcv.setter
    def vcv(self, x):
        s = np.shape(x)
        if s[0] != s[1]:
            raise ValueError('mvo vcv not symetrical; shape {}'.format(s))
        elif not all(i >=0 for i in np.diag(x)):
            raise ValueError('vcv input with -ve variances; {}'.format(np.diag(x)))
        self.__vcv = x
        
    def _constraint_dict_append(self, const_dic):
        """ Helper function to define constraints.
        Input is a dict of the form:
            {'name':([universe], operator, value)} for example
            {'Max_Equity':([Equity], lessthan, 0.5)} """
        
        for key, item in const_dic.items():
            
            # Get asset group, operator & constraint value; ensure group is list
            group, operator, value = item
            group = list([group]) if type(group) is not list else group
            
            # CVXPY uses numpy not pandas so need to pull indices of assets
            if isinstance(group[0], int): index = group
            elif isinstance(group[0], str):
                index = [i for i, x in enumerate(self.universe) if x in group]
            
            # Build constraint using operator
            if operator in ['equal','=','==']:
                self.constraints.append(sum(self.w[index]) == value)
            elif operator in ['lessthan', 'less', '<=', '<']:
                self.constraints.append(sum(self.w[index]) <= value)
            elif operator in ['eachless', 'capped', 'cap']:
                self.constraints.append(self.w[index] <= value)
            elif operator in ['greater', '>=', '>']:
                self.constraints.append(sum(self.w[index]) >= value)
            elif operator in ['eachgreater', 'floored', 'floor']:
                self.constraints.append(sum(self.w[index]) >= value)
            elif operator == 'between':
                self.constraints.append(sum(self.w[index]) >= value[0])
                self.constraints.append(sum(self.w[index]) <= value[1])
                
    def _set_cvx_risk_rtn_params(self):
        """ CVXPY uses fixed values of mu & vcv when risk & return are defined.
        Therefore we MUST update if input E(R) or VCV change """  
        self.rtn = self.mu.T * self.w
        self.risk = cvx.quad_form(self.w, self.vcv)
        self.constraints = []
        self._constraint_dict_append(self.asset_constraints)
        [self.constraints.append(i) for i in [sum(self.w)==1, self.w >= 0]]
        

    def min_vol(self):
        """ Optimise to find the Minimum Volatility Portfolio """
                
        self._set_cvx_risk_rtn_params()        # setup risk & rtn formula
        
        # Setup CVXPY problem and solve for minimum variance
        objective = cvx.Minimize(self.risk)
        cvx.Problem(objective, self.constraints).solve()
        return self.w.value.round(4), cvx.sqrt(self.risk).value, self.rtn.value
    
    def target_vol(self, vol=0):
        """ Optimise to Maximise Return subject to volatility constraint. 
        In the absence of a vol target will seek to maximise return """
        
        self._set_cvx_risk_rtn_params()    # setup risk & rtn formula
        
        # Update Constraints - Important that this includes the VOL TARGET
        constraints = self.constraints.copy()
        #[constraints.append(i) for i in [sum(self.w)==1, self.w >= 0]]
        if vol > 0: constraints.append(self.risk<=(np.float64(vol)**2))
        
        # setup CVXPY problem and solve for population mean
        objective = cvx.Maximize(self.rtn)
        cvx.Problem(objective, constraints).solve()
        return self.w.value.round(4), cvx.sqrt(self.risk).value, self.rtn.value
    
    def risk_aversion(self, gamma=0):
        """ Optimisation using a Risk-Aversion parameter
        Solve: Maximise[Return - Risk] = [w * mu - (gamma/2)* w * vcv * w.T]
        """

        self._set_cvx_risk_rtn_params()    # setup risk & rtn formula
        self.gamma.value = gamma
        
        # setup CVXPY problem and solve for max rtn given risk aversion level
        objective = cvx.Maximize(self.rtn - (gamma/2) * self.risk)
        cvx.Problem(objective, self.constraints).solve()    # Solve & Save Output
        return self.w.value.round(4), cvx.sqrt(self.risk).value, self.rtn.value

    def frontier(self, method='vol_linspace', **kwargs):
        """
        Mean Variance Optimisation across Efficient Frontier
        
        Frontier can either be calculated using a range of vol targets or by
        using a range of risk aversion paramenters.
        
        INPUTS:
            method - 'vol_linspace'(default) | 'target_vol' | 'risk_aversion'
        
        KWARGS:
            gamma_rng - uppper & lower bounds for vol (or risk aversion parameter)
            steps - vol steps (ie 0.25%) or no of intervals in logspace for risk_aversion
        """
        
        # Dummy Output DataFrames
        wgts = pd.DataFrame(index=self.universe)
        stats = pd.DataFrame(index=['rtn', 'vol'])
        
        if method == 'vol_linspace':
            OPTIMISE = lambda x: self.target_vol(x)    
            gmin, gmax = self.min_vol()[1], self.target_vol()[1]
            steps = kwargs['steps'] if 'steps' in kwargs else 10
            gamma = np.linspace(gmin, gmax, num=steps, endpoint=True)    
        
        elif method == 'target_vol':
            OPTIMISE = lambda x: self.target_vol(x)
            if 'gamma_rng' in kwargs: gmin, gmax = kwargs['gamma_rng']
            else: gmin, gmax = self.min_vol()[1], self.target_vol()[1]
            steps = kwargs['steps'] if 'steps' in kwargs else 0.0025
            gamma = np.arange(gmin, gmax, steps)
        
        elif method == 'risk_aversion':
            OPTIMISE = lambda x: self.risk_aversion(x)
            gmin, gmax = kwargs['gamma_rng'] if 'gamma_rng' in kwargs else [-1, 5]
            steps = kwargs['steps'] if 'steps' in kwargs else 101
            gamma = np.logspace(gmin, gmax, steps)[::-1]
        
        else:
            raise ValueError('ERR: {} NOT valid optimisation method'.format(method))
        
        for g in gamma:
            
            x = OPTIMISE(g)    # Optimise using lamda func declared by method
            
            # Test optimiser solution before adding to output
            # ignore output if no solution available - typically vol too low
            # ignore where soln vol < tgt vol; optimiser returns max achievable vol.
            if x[1] == None: continue
            #elif opt_type == 'target_vol' and abs(g-x[1]) > steps: break  
    
            wgts["{:.4f}".format(g)] = x[0]             # add weights vector
            stats["{:.4f}".format(g)] = [x[2], x[1]]    # add risk & return stats 
        
        # OUTPUT
        self.port_weights = wgts.round(4)
        self.port_stats = stats.round(4).astype(float)
        
        return self.port_weights, self.port_stats
    
# %% OPTIMISE OVER EFFICIENT FRONTIER  
        
def mvo_frontier(mu, vcv, constraints=None, method='vol_linspace', **kwargs):
    """ Mean Variance Efficient Frontier
    - One liner func to setup optimisation, solve over frontier & output data
    - Can be achieved with 4 lines of code & mvo class """
    opt = mvo(mu=mu, vcv=vcv, asset_constraints=constraints)
    opt.frontier(method=method, **kwargs)    
    
    return opt.port_weights, opt.port_stats.astype(float)

# %% RESAMPLED FRONTIER FUNCTIONS
    
def resampled_frontier(mu, vcv, constraints=None, seed=10, steps=20):
    """ Resampled Efficient Frontier
    
    There are several methods to achieve, all with pros & cons. Here we:
        * generate efficient frontier given a 1xM vector of expected returns mu 
        * find min vol. portfolio & max return portfolios
        * find K portfolios between min vol & max rtn, where K is steps
        * gives us a matrix, w0, of size KxM (portfolios by asset weights)
        * iterate using statistically equivalent expected returns
        * find w1 as mean KxM matrix from all iterations
        * Recalculate risk return statistics for w0 & w1
    
    Mechanically a relatively simple solution. Each iteration will yield a KxM 
    matrix making the averaging across frontiers easy. However we won't hit 
    each vol level on each frontier meaning the average of each portfolio Ki
    won't have the same vol as K0. This tends to lead to shorter frontiers 
    where higher vol solutions aren't attainable. Another issue is that we don't
    have fixed vol points on the frontier which can be annoying.
    
    INPUTS:
        mu - expected returns as pd.DataFrame with asset names as index
        vcv - covariance matrix
        constraints - dictionary of asset class level constraints
        seed - No of Monte-Carlo simulations
        steps - intervals between min vol & max rtn portfolios
        
    OUTPUT:
        w0 - pd.DataFrame of weights of the "true frontier"
        w1 - resampled DataFrame of weights
        stats - risk & rtn stats for w0 & w1
    """
    
    i, no_soln = (0, 0)
    while i <= seed:       

        # 1st pass setup optimisation; then update E(Rtns)
        if i == 0: opt = mvo(mu=mu, vcv=vcv, asset_constraints=constraints)
        else: opt.mu = np.random.multivariate_normal(mu, vcv)
        
        try:
            # Optimise across frontier & update columns to integer index 
            # 1st pass setup "true frontier" & then the resampled wgts array
            w, _ = opt.frontier(opt_type='vol_linspace', steps=steps)
            w.columns = list(range(steps))
            if i == 0: w0, w1 = [w] * 2
            else: w1 = w1 + w
            i += 1          # update iterator
        except:
            # Re-run if updated E(R) can't find a solution; limit re-runs
            if no_soln == seed:
                raise ValueError('Monte-Carlo simulationss ended after {} failed solutions'.format(seed))
            else: no_soln += 1
            
    w1 = w1 / (seed+1)    # divide sum by seed (to find average weight)
    rtnrisk = lambda w: [w @ mu.T, np.sqrt(w @ vcv @ w.T), ]    # @ is matmult
    stats = pd.DataFrame(columns=w0.columns, index=['rtn0', 'vol0', 'rtn1', 'vol1'])
    
    # Populate Stats Frame
    for c in stats:
        stats.loc[:,c].iloc[:2] = rtnrisk(w0.loc[:,c])
        stats.loc[:,c].iloc[2:] = rtnrisk(w1.loc[:,c])
            
    return w0.round(4), w1.round(4), stats.astype(float)


# %% TEST FUNCTION
    
def _test_function():

    # Dummy Data Set
    data = pd.DataFrame(index=['rtns', 'vol'])
    data['equity'] = [0.10, 0.15]
    data['credit'] = [0.08, 0.1]
    data['rates'] = [0.05, 0.06]
    data['cash'] = [0.01, 0.001]
    
    # Expected Returns
    mu = data.loc['rtns',:]
    
    # VCV
    vcv = np.eye(len(data.loc['rtns',:]))
    np.fill_diagonal(vcv, data.loc['vol',:] ** 2) 
    
    # Asset Constraints
    #ac = {'FixEq':(['equity', '>=', 0.20]), 'MaxCash':(['cash', '==', 0.01])}
    
    #w0, w1, stats = resampled_frontier(mu=mu, vcv=vcv, constraints=ac, seed=100)
    
    #plt.plot(stats.T['vol0'], stats.T['rtn0'])
    #plt.plot(stats.T['vol1'], stats.T['rtn1'])
    
    return mu   #w0, w1, stats