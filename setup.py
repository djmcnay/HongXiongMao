#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: D J McNay
Created on Sat Jun 15 - loitering at Blackbird Bakery in East Dulwich

First attempt at creating a pip installable repo from github
"""

#from setuptools import setup
from setuptools import setup, find_packages

setup(
    name = 'badgertools',
    version = '0.1dev',
    packages=find_packages(),
    
    # Required Dependencies
    install_requires=['pandas',
                      'numpy',
                      'plotly',
                      'quandl',
                      'alpha_vantage',
                      ],
                      
    # Optional Dependencies
    # Still need to work on this

    # Meta-Data
    author = 'David J McNay',
    author_email = 'djmcnay@gmail.com',
    license = 'MIT',
    description='test packaging',
    long_description=open('README.md').read(),
)