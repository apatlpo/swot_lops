#!/usr/bin/env python

from distutils.core import setup

INSTALL_REQUIRES = ['xarray >= 0.10.6', ]

setup(name='swotlib',
      description='SWOT analysis utils',
      url='https://github.com/apatlpo/swot_lops',
      packages=['swotlib'])

#      install_requires=INSTALL_REQUIRES,


