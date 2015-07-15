#!/usr/bin/env python

from setuptools import setup
import thunder

setup(
    name='thunder-python',
    version=str(thunder.__version__),
    description='Large-scale neural data analysis in Spark',
    author='The Freeman Lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/thunder-project/thunder',
    packages=['thunder',
              'thunder.clustering',
              'thunder.decoding',
              'thunder.factorization',
              'thunder.extraction',
              'thunder.extraction.block',
              'thunder.extraction.feature',
              'thunder.extraction.block.methods',
              'thunder.extraction.feature.methods',
              'thunder.imgprocessing',
              'thunder.imgprocessing.regmethods',
              'thunder.lib',
              'thunder.rdds',
              'thunder.rdds.fileio',
              'thunder.rdds.imgblocks',
              'thunder.regression',
              'thunder.regression.linear',
              'thunder.regression.nonlinear',
              'thunder.standalone',
              'thunder.utils',
              'thunder.utils.data',
              'thunder.viz'],
    scripts = ['bin/thunder', 'bin/thunder-submit', 'bin/thunder-submit-example', 'bin/thunder-ec2'],
    package_data = {'thunder.utils': ['data/fish/series/conf.json', 'data/fish/series/*.bin', 'data/fish/images/*.tif', 'data/iris/conf.json', 'data/iris/iris.bin', 'data/iris/iris.mat', 'data/iris/iris.npy', 'data/iris/iris.txt', 'data/mouse/images/conf.json', 'data/mouse/images/*.bin', 'data/mouse/params/covariates.json', 'data/mouse/series/conf.json', 'data/mouse/series/*.bin'], 'thunder.lib': ['thunder_2.10-' + str(thunder.__version__) + '.jar', 'thunder_python-' + str(thunder.__version__) + '-py2.7.egg']},
    long_description=open('README.rst').read(),
    install_requires=open('requirements.txt').read().split()
)
