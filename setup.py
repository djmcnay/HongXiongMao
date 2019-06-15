#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: D J McNay
Created on Sat Jun 15 - loitering at Blackbird Bakery in East Dulwich

First attempt at creating a pip installable repo from github
"""

from setuptools import setup

setup(
    name = 'badgertools',
    version = '0.1dev',
    packages=['badgertools'],
    install_requires=['plotly'],
    author = 'David J McNay',
    author_email = 'djmcnay@gmail.com',
    license = 'MIT',
    description='test packaging',
    long_description=open('README.md').read(),
)
