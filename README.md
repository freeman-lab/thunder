# thunder

[![Latest Version](https://img.shields.io/pypi/v/thunder-python.svg?style=flat-square)](https://pypi.python.org/pypi/thunder-python)
[![Build Status](https://img.shields.io/travis/thunder-project/thunder/master.svg?style=flat-square)](https://travis-ci.org/thunder-project/thunder) 
[![Gitter](https://img.shields.io/gitter/room/thunder-project/thunder.svg?style=flat-square)](https://gitter.im/thunder-project/thunder)

> scalable analysis of image and time series analysis in python

Thunder is an ecosystem of tools for the analysis of image and time series data in Python. It provides data structures and algorithms for loading, processing, and analyzing these  data, and can be useful in a variety of domains, including neuroscience, medical imaging, video processing, and geospatial and climate analysis. It can be used locally, but also supports large-scale analysis through the distributed computing engine [`spark`](https://github.com/apache/spark). All data structures and analyses in Thunder are designed to **run identically and with the same API** whether local or distributed.

Thunder is designed around modularity and composability — the core `thunder` package, in this repository, only defines common data structures and read/write patterns, and most functionality is broken out into several related packages. Each one is independently versioned, with its own GitHub repository for organizing issues and contributions. 

This readme provides an overview of the core `thunder` package, its data types, and methods for loading and saving. Tutorials, detailed API documentation, and info about all associated packages can be found at the [documentation site](http://docs.thunder-project.org).

## install

The core `thunder` package defines data structures and read/write patterns for images and series data. It is built on [`numpy`](https://github.com/numpy/numpy), [`scipy`](https://github.com/scipy/scipy), [`scikit-learn`](https://github.com/scikit-learn/scikit-learn), and [`scikit-image`](https://github.com/scikit-image/scikit-image), and is compatible with Python 2.7+ and 3.4+. You can install it using:

```
pip install thunder-python
```

## related packages

Lots of functionality in Thunder, especially for specific types of analyses, is broken out into the following separate packages. 

- [`thunder-regression`](https://github.com/thunder-project/thunder-regression) mass univariate regression algorithms
- [`thunder-factorization`](https://github.com/thunder-project/thunder-factorization) matrix factorization algorithms 
- [`thunder-registration`](https://github.com/thunder-project/thunder-registration) registration for image sequences

You can install the ones you want with `pip`, for example

```
pip install thunder-regression
pip install thunder-registration
```

## example

Here's a short snippet showing how to load an image sequence (in this case random data), median filter it, transform it to a series, detrend and compute a fourier transform on each pixel, then convert it to an array.

```python
import thunder as td

data = td.images.fromrandom()
ts = data.median_filter(3).toseries()
frequencies = ts.detrend().fourier(freq=3).toarray()
```

## usage

Most workflows in Thunder begin by loading data, which can come from a variety of sources and locations, and can be either local or distributed (see below).

The two primary data types are `images` and `series`. `images` are used for collections or sequences of images, and are especially useful when working with movie data. `series` are used for collections of one-dimensional arrays, often representing time series.

Once loaded, each data type can be manipulated through a variety of statistical operators, including simple statistical aggregiations like `mean` `min` and `max` or more complex operations like `gaussian_filter` `detrend` and `subsample`. Both `images` and `series` objects are wrappers for ndarrays: either a local [`numpy`](https://github.com/numpy/numpy) `ndarray` or a distributed ndarray using [`bolt`](https://github.com/bolt-project/bolt) and [`spark`](https://github.com/apache/spark). Calling `toarray()` on an `images` or `series` object at any time returns a local [`numpy`](https://github.com/numpy/numpy) `ndarray`, which is an easy way to move between Thunder and other Python data analysis tools, like [`pandas`](https://github.com/pydata/pandas) and [`scikit-learn`](https://github.com/scikit-learn/scikit-learn).

For a full list of methods on `image` and `series` data, see the [documentation site](http://docs.thunder-project.org).

## loading data

Both `images` and `series` can be loaded from a variety of data types and locations. For all loading methods, the optional argument `engine` allows you to specify whether data should be loaded in `'local'` mode, which is backed by a `numpy` array, or in `'spark'` mode, which is backed by an RDD.

All loading methods are available on the module for the corresponding data type, for example

```python
import thunder as td

data = td.images.fromtif('/path/to/tifs')
data = td.series.fromarray(somearray)
data_distributed = ts.series.fromarray(somearray, engine=sc)
```

The argument `engine` can be either `None` for local use or a `SparkContext` for distributed use with Spark. And in either case, methods that load from files e.g. `fromtif` or `frombinary` can load from either a local filesystem or Amazon S3, with the optional argument `credentials` for S3 credentials. See the [documentation site](http://docs.thunder-project.org) for a full list of data loading methods.

## using with spark

Thunder doesn't require Spark and can run locally without it, but Spark and Thunder work great together! To install and configure a Spark cluster, consult the official [Spark documentation](http://spark.apache.org/docs/latest). Thunder supports Spark version 1.5+, and uses the Python API PySpark. If you have Spark installed, you can install Thunder just by calling `pip install thunder-python` on both the master node and all worker nodes of your cluster. Alternatively, you can clone this GitHub repository, and make sure it is on the `PYTHONPATH` of both the master and worker nodes. 

Once you have a running cluster with a valid `SparkContext` — this is created automatically as the variable `sc` if you call the `pyspark` executable — you can pass it as the `engine` to any of Thunder's loading methods, and this will load your data in distributed `'spark'` mode. In this mode, all operations will be parallelized, and chained operations will be lazily executed.

## contributing

Thunder is a community effort! The codebase so far is due to the excellent work of the following individuals:

> Andrew Osheroff, Ben Poole, Chris Stock, Davis Bennett, Jascha Swisher, Jason Wittenbach, Jeremy Freeman, Josh Rosen, Kunal Lillaney, Logan Grosenick, Matt Conlen, Michael Broxton, Noah Young, Ognen Duzlevski, Richard Hofer, Owen Kahn, Ted Fujimoto, Tom Sainsbury, Uri Laseron

If you run into a problem, have a feature request, or want to contribute, submit an issue or a pull request, or come talk to us in the [chatroom](https://gitter.im/thunder-project/thunder)!
