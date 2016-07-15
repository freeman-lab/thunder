#!/usr/bin/env python

from setuptools import setup

version = '1.2.0'

setup(
    name='thunder-python',
    version=version,
    description='large-scale image and time series analysis',
    author='freeman-lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/thunder-project/thunder',
    packages=[
        'thunder',
        'thunder.blocks',
        'thunder.series',
        'thunder.images'
    ],
    install_requires=open('requirements.txt').read().split('\n'),
    long_description='See https://github.com/thunder-project/thunder'
)
