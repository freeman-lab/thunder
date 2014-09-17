.. _basic_usage:

Basic usage
===========

thunder implements many common neuroscience analyses and workflows. These are designed to support both modular, interactive anlaysis, as well as standalone pipelines for common routines. We'll walk through a very simple example: computing the standard deviation of each of many neural signals from imaging data. 

For an interactive analysis, we first start the shell

.. code-block:: bash

	thunder

Import the functions and classes we'll need, in this case ``Stats``.

.. code-block:: python

	>> from thunder.timeseries import Stats

First we load some toy example data

.. code-block:: python

	>> data = tsc.loadExample("fish")

``tsc`` is a modified Spark Context, created when you start thunder, that serves as an entry point for loading distributed datasets. Once the data is loaded, use the stats class to compute the standard deviation

.. code-block:: python

	>> stat = Stats("std")
	>> vals = stat.calc(data)

At this point, we could look at the result with matplotlib

.. code-block:: python

	>> import matplotlib.pyplot as plt
	>> from thunder.utils import pack
	>> img = pack(vals)
	>> plt.imshow(img)
	>> plt.show()

If you do the rest of your analysis in Matlab, it's easy to save to a MAT file

.. code-block:: python

	>> from thunder.utils import save
	>> save(vals, "test", "stats", "matlab")

This will put a MAT file called ``stats`` in the folder ``test`` in your current directory. 

The exact same analysis can easily be run as a standalone job directly from the terminal

.. code-block:: bash

	thunder-submit timeseries/stats <path/to/data> <path/to/output> "std"









