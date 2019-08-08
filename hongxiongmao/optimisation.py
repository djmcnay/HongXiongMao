#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HongXiongMao Optimisation Functions


"""

import numpy as np
import pandas as pd
import cvxpy as cvx

# %% MEAN VARIANCE OPTIMISER

class mvo(object):
    """
    Mean Variance Optimisation Class
    
    May be used as a freestanding MVO optimiser but designed to be subclass.
    Other packages available but this uses the CVXPY for convex optimisation.
    
    BASIC FUNCTION FLOW:
        1. initialise with expected returns, risk model
        2. run preop_setup which is a preoptimisation setup in cvxpy & includes
           adding asset level constraints
        3. run selected optimsation type - including specific constraints and
           parameters - such as gamma (risk aversion) or vol. target
    
    REQUIRED INPUTS:
        mu - pd.Series or np array of expected returns
        vcv - variance covariance matrix. Module will do some limited checks
              such as symmetary & no -ve variances but NOT fully error checked
        
        These can either be initisalise when creating the class object or you
        can add set them later
            
    OPTIMISATION TYPES:
        1. risk_aversion - Max[return - (aversion/2) * risk]
        2. target_vol = Max[return] subject to vol. constraint
        3. tracking_error - TO BE BUILT
        
        NB/ For more details look at the specific module info
    
    """
    
    def __init__(self, mu=None, vcv=None):
        if mu is not None: self.mu = mu
        if vcv is not None: self.vcv = vcv
        
    # Expected Returns
    # Ensure vector, if Pandas then save Index as universe attribute
    @property
    def mu(self): return self.__mu
    @mu.getter
    def mu(self): return self.__mu.value
    @mu.setter
    def mu(self, x):
        if len(np.shape(x)) != 1:raise ValueError('mvo mu input {} is non-vector'.format(x))
        if isinstance(x, pd.Series):
            self.universe = list(x.index)
            self.__mu = cvx.Parameter(len(x), value=x.values)
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
        """
        Helper function to define constraints
        
        Input is a dict of the form:
            {'name':([universe], operator, value)} for example
            {'Max_Equity':([Equity], lessthan, 0.5)}
        
        Each key is converted to cvxpy constraint & appended to constraints list
        """
        
        # iterate through each item in constraint dict
        for key, item in const_dic.items():
            
            # Extract asset group, operator and constraint value from item
            group, operator, value = item
            
            # Ensure universe group is a list
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
                
    def set_cvx_risk_rtn_params(self):
        """
        Update CVXPY Risk & Return formula
        These are fixed with current mu & vcv values when defined.
        Therefore update between optimisations if input E(R) or VCV changed
        """
        self.rtn = self.mu.T * self.w
        self.risk = cvx.quad_form(self.w, self.vcv)
        
        return

    def preop_setup(self, asset_constraints=None):
        """
        Helper function to setup parts of CVXPY optimisation
    
        Does a number of things:
            - creates vector variable for port weights in optimser
            - initialises basic constriants (no shorting, sum to 100)
            - allows addition of asset_class constraints
            - builds return function in cvxpy style
            - builds risk (variance) in cvxpy style
        
        Mostly these are the parts of the optimisation problem that only needs
        to be done once. If we are only running one problem them we can run
        preop_setup as part of the optimisation. But if you are doing multiple
        optimisations i.e. changing gamma (risk aversion) across the curve then
        it would be better to run this first and set preop=False when running
        risk_aversion() or target_vol()
        
        NB/ Changing expected returns REQUIRES re-preop'ing
        """
        
        # vector of portfolio weights
        n = len(self.mu)
        self.w = cvx.Variable(n)  
        
        # Set risk & return formula for optimisation engine
        self.set_cvx_risk_rtn_params()
        
        # General constraints
        self.constraints = [sum(self.w)==1, self.w >= 0]
        
        # Asset Class constraints
        if isinstance(asset_constraints, dict):
            self._constraint_dict_append(asset_constraints)
        
        # risk aversion parameter (even if we don't need it)
        self.gamma = cvx.Parameter(nonneg=True) 
        
    def risk_aversion(self, gamma=0, ass_const=None, preop=False):
        """
        Optimisation using a Risk-Aversion parameter
        
        Solve:
            Maximise[Return - Risk] = [w * mu - (gamma/2)* w * vcv * w.T]
        """
        
        if preop: self.preop_setup(ass_const)
        
        # Additional constraints
        constraints = self.constraints.copy()
        self.gamma.value = gamma
        
        # Setup optimisation problem
        objective = cvx.Maximize(self.rtn - (gamma/2) * self.risk)
        problem = cvx.Problem(objective, constraints)
        problem.solve()    # Solve & Save Output

        return self.w.value, cvx.sqrt(self.risk).value, self.rtn.value
    
    def target_vol(self, vol=0.10, ass_const=None, preop=False):
        """
        Optimise to Maximise Return subject to volatility constraint
        """

        if preop: self.preop_setup(ass_const)
        
        # Update Constraints - Important that this includes the VOL TARGET
        constraints = self.constraints.copy()
        constraints.append(self.risk<=(np.float64(vol)**2))
        
        # setup CVXPY problem and solve for population mean
        objective = cvx.Maximize(self.rtn)
        problem = cvx.Problem(objective, constraints)
        problem.solve()
        
        return self.w.value, cvx.sqrt(self.risk).value, self.rtn.value
    
    def frontier(self, ass_const=None, opt_type='target_vol', preop=True, **kwargs):
        """
        Mean Variance Optimisation across Efficient Frontier
        
        Frontier can either be calculated using a range of vol targets or by
        using a range of risk aversion paramenters.
        
        INPUTS:
            opt_type -  'target_vol'(default) | 'risk_aversion'
        
        KWARGS:
            gamma_rng - uppper & lower bounds for vol (or risk aversion parameter)
            steps - vol steps (ie 0.25%) or no of intervals in logspace for risk_aversion
        """
        
        if preop: self.preop_setup(ass_const)
        
        # Dummy Output DataFrames
        wgts = pd.DataFrame(index=self.universe)
        stats = pd.DataFrame(index=['rtn', 'vol'])
        
        ### Select optimisation type - either target vol points or a risk aversion
        if opt_type == 'target_vol':
            OPTIMISE = lambda x: self.target_vol(x)
            gmin, gmax = kwargs['gamma_rng'] if 'gamma_rng' in kwargs else [0, 0.20]
            steps = kwargs['steps'] if 'steps' in kwargs else 0.0025
            gamma = np.arange(gmin, gmax, steps)    # Fix Iteration Range
        elif opt_type == 'risk_aversion':
            OPTIMISE = lambda x: self.risk_aversion(x)
            gmin, gmax = kwargs['gamma_rng'] if 'gamma_rng' in kwargs else [-1, 5]
            steps = kwargs['steps'] if 'steps' in kwargs else 101
            gamma = np.logspace(gmin, gmax, steps)[::-1]
         
        ### Iterate through each trial
        for g in gamma:
            
            # Optimise using lamda function declared by optimisation type        
            x = OPTIMISE(g)
            
            # Test optimiser solution before adding to output
            # ignore output if no solution available - typically vol too low
            # ignore where soln vol < tgt vol; optimiser returns max achievable vol.
            if x[1] == None: continue
            #elif opt_type == 'target_vol' and abs(g-x[1]) > steps: break  
    
            wgts["{:.4f}".format(g)] = x[0]             # add weights vector
            stats["{:.4f}".format(g)] = [x[2], x[1]]    # add risk & return stats 
        
        ## OUTPUT
        self.port_weights = wgts.round(4)
        self.port_stats = stats.round(4).astype(float)
        
        return self.port_weights, self.port_stats
        
# %% OPTIMISE OVER EFFICIENT FRONTIER  
        
def mvo_frontier(mu, vcv, constraints=None, opt_type='target_vol', **kwargs):
    """
    Mean Variance Efficient Frontier
    - One liner func to setup optimisation, solve over frontier & output data
    - Can be achieved with 4 lines of code & mvo class """
    
    opt = mvo(mu=mu, vcv=vcv)
    opt.preop_setup(asset_constraints=constraints)
    opt.frontier(opt_type=opt_type, **kwargs)    
    
    return opt.port_weights, opt.port_stats.astype(float)

# %% RESAMPLED FRONTIER FUNCTIONS
    
def resampled_frontier(mu, vcv, constraints=None, seed=100, steps=51, **kwargs):
    """
    Resampled Efficient Frontier
    
    There are a few methods to achieve this - here we average port_weights from
    whole frontier & average at each gamma (risk aversion). This is relatively
    simple but means we need to calibrate gamma_rng and we aren't guaranteed
    a portfolio at all vol. points we maybe interest in.
    
    WARNING: SIGNIFICANT RUN-TIME
    """
    
    ## Setup Optimisation Problem
    opt = mvo(mu=mu, vcv=vcv)
    opt.preop_setup(asset_constraints=constraints)
    
    i = 0
    while i < seed:
        
        # Update expected returns with statistically equivalent returns
        # Then find solution for updated Expected Returns
        if i > 0: opt.mu = np.random.multivariate_normal(mu, vcv)
        w, s = opt.frontier(opt_type='risk_aversion', steps=steps, **kwargs)
        
        if i == 0: w0, w1 = w, w    # 1st pass we save "true" frontier
        else: w1 = w1 + w           # later passes SUM wgts for each gamma
        
        i += 1    # Update iteratior
    
    ## Post-Op
    w1 = w1 / seed    # divide sum by seed (to find average weight)
    rtnrisk = lambda w: [w @ mu.T, np.sqrt(w @ vcv @ w.T), ]    # @ is matmult
    stats = pd.DataFrame(columns=w0.columns, index=['rtn0', 'vol0', 'rtn1', 'vol1'])
    
    # Populate Stats Frame
    for c in stats:
        stats.loc[:,c].iloc[:2] = rtnrisk(w0.loc[:,c])
        stats.loc[:,c].iloc[2:] = rtnrisk(w1.loc[:,c])
    
    return w0, w1, stats.astype(float)

def resampled_vol_tgt(mu, vcv, constraints=None, seed=10, steps=0.01, **kwargs):
    """
    Resampled Efficient Frontier
    There are several methods to achieve this - here we iterate through each vol
    point and trial statistically equivalent expected returns for that vol level.
    One    
    
    """
    
    ## Setup Optimisation Problem
    opt = mvo(mu=mu, vcv=vcv)
    opt.preop_setup(asset_constraints=constraints)
    
    gamma_rng = np.arange(0.14, 0.16, 0.01)
    
    # iterate through each vol point in range
    for g in gamma_rng:
        
        # Now resample for the g'th vol point
        i = 0
        while i < seed:
            
            # update expected returns & run optimiser
            opt.mu = np.random.multivariate_normal(mu, vcv)
            opt.set_cvx_risk_rtn_params()
            x = opt.target_vol(g)
            
            print(x[1])
            i += 1
    
    
    
    #while i < seed:
    #i = 0
    #    if i > 0: opt.mu = np.random.multivariate_normal(mu, vcv)
    #    w, s = opt.frontier(opt_type='target_vol', steps=steps, **kwargs)
    #    
    #    print(s.loc['vol',:].iloc[-1])
    #    
    #    i += 1
    
    return

# %%
        
data = pd.DataFrame(index=['rtns', 'vol'])
data['equity'] = [0.06, 0.15]
data['credit'] = [0.04, 0.1]
data['rates'] = [0.025, 0.06]
data['cash'] = [0.01, 0.001]

vcv = np.eye(len(data.loc['rtns',:]))
np.fill_diagonal(vcv, data.loc['vol',:] ** 2)

# %%        
ac = {'FixEq':(['equity', '==', 0.15])}
wgts, stats = mvo_frontier(vcv=vcv, mu=data.loc['rtns',:], opt_type='risk_aversion')
#opt = mvo(mu=data.loc['rtns',:], vcv=vcv)
#wgts, stats = opt.frontier(preop=True, opt_type='risk_aversion', ass_const=ac)

#w0, w1, stats = resampled_frontier(mu=data.loc['rtns',:], vcv=vcv, steps=100, seed=10, gamma_rng=[-1, 20])
x = resampled_vol_tgt(mu=data.loc['rtns',:], vcv=vcv, seed=12)


#import matplotlib.pyplot as plt
#plt.plot(stats.T['vol0'], stats.T['rtn0'])
#plt.plot(stats.T['vol1'], stats.T['rtn1'])