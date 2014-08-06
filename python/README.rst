.. figure:: https://travis-ci.org/freeman-lab/thunder.png
   :align: left
   :target: https://travis-ci.org/freeman-lab/thunder

Thunder
=======

Large-scale neural data analysis with Spark - `project page`_

.. _project page: http://freeman-lab.github.io/thunder/

About
-----

Thunder is a library for analyzing large-scale neural data. It's fast to run, easy to develop for, and can be used interactively. It is built on Spark, a new framework for cluster computing.

Thunder includes utilties for data loading and saving, and modular functions for time series statistics, matrix decompositions, and fitting algorithms. Analyses can easily be scripted or combined. It is written in Spark's Python API (Pyspark), making use of scipy, numpy, and scikit-learn. Experimental streaming analyses are availiable in Scala, and we plan to port some functionality to Scala in the future for improved performance.

Quick start
-----------

Thunder is designed to run on a cluster, but local testing is a great way to learn and develop. Many computers can install it with just a few simple steps. If you aren't currently using Python for scientific computing, `Anaconda`_ is highly recommended.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/

1) Download the latest, pre-built version of `Spark`_, and set one environmental variable

.. _Spark: http://spark.apache.org/downloads.html

::

	export SPARK_HOME=/your/path/to/spark

2) Install Thunder

:: 

	pip install thunder-python

3) Start Thunder from the terminal

:: 

	thunder
	>> from thunder.utils import DataSets
	>> from thunder.factorization import ICA
	>> data = DataSets.make(sc, "ica")
	>> model = ICA(c=2).fit(data)

To run in iPython, just set this environmental variable before staring:

::

	export IPYTHON=1

To run analyses as standalone jobs, use the submit script

::

	thunder-submit <package/analysis> <datadirectory> <outputdirectory> <opts>

We also include a script for launching an Amazon EC2 cluster with Thunder presintalled

::

	thunder-ec2 -k mykey -i mykey.pem -s <number-of-nodes> launch <cluster-name>


Analyses
--------

Thunder currently includes five packages: classification, clustering, factorization, regression, and timeseries, as well as an io package for loading and saving (see Input format and Output format), and a util package for utilities (like common matrix operations). Packages include scripts for running standalone analyses, but the underlying classes and functions can be used from within the PySpark shell for easy interactive analysis.

Input and output
----------------

Thunder is built around a commmon input format for raw neural data: a set of signals as key-value pairs, where the key is an identifier, and the value is a response time series. In imaging data, for example, each record would be a voxel or an ROI, the key an xyz coordinate, and the value a flouresence time series. This is a useful representation because most analyses parallelize across neural signals (i.e. across records). 

These key-value records can, in principle, be stored in a variety of cluster-accessible formats, and it does not affect the core functionality (besides loading). Currently, the loading function assumes a text file input, where the rows are neural signals, and the columns are the keys and values, each number separated by space. Support for flat binary files is coming soon.

All metadata (e.g. parameters of the stimulus or behavior for regression analyses) can be provided as numpy arrays or loaded from MAT files, see relavant functions for more details.

Results can be visualized directly from the python shell ir iPython notebook, or saved as MAT files, text files, or images.

Road map
----------------
If you have other ideas or want to contribute, submit an issue or pull request!

- New file formats for input data
- Automatic extract-transform-load for more raw formats (e.g. raw images)
- Analysis-specific visualizations
- Unified metadata representation
- Port versions of most common workflows to scala
