#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
import cvxpy as cvx

class mvo(object):
    """
    """
    
    def __init__(self, **kwargs):
        self.name = kwargs['name'] if 'name' in kwargs else 'optimisation'
        return
    