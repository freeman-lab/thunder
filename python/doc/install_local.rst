.. _install_local_ref:

Installation
============

thunder can be installed on most modern computers running Python 2.7. It has only been tested on Mac OS X, but should also work on Windows and Linux. If you aren't already using scientific Python, we recommend getting the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_.

To follow along with the instructions below, you'll just need a command line (e.g. the Terminal app in Mac OS X), a web browser, and the ``pip`` Python package manager (which is included with Anaconda). These instructions get Spark and thunder running on a local machine. That's great for learning the library and developing analyses. But thunder is meant to run on a cluster. Once you install, the command ``thunder-ec2`` lets you launch a cluster on Amazon's EC2, see :ref:`install_ec2_ref`.

Installing Spark 
~~~~~~~~~~~~~~~~
First you need a working installation of Spark. You can `download <http://spark.apache.org/downloads.html>`_ one of the pre-built versions (pick the one labeled Hadoop 2), or you can download the sources, and follow `these instructions <http://spark.apache.org/docs/latest/building-with-maven.html>`_ to build from source.

Once you have downloaded Spark, set an environmental variable by typing the following into the terminal (here we assume you downloaded a pre-built version and put it in your downloads folder)

.. code-block:: bash

	export SPARK_HOME=~/downloads/spark-1.0.2-bin-hadoop2

To make this setting permanent, add the above line to your bash profile (usually located in ``~/.bash_profile`` on Mac OS X), and open a new terminal so the change takes effect. Otherwise, you'll need to enter this line during each terminal session.

Installing thunder
~~~~~~~~~~~~~~~~~~
Type the following line into the terminal

.. code-block:: bash
	
	pip install thunder-python

If it didn't work, try

.. code-block:: bash
	
	sudo pip install thunder-python

If the installation was successful you will have three new commands available: ``thunder``, ``thunder-submit``, and ``thunder-ec2``. You can check the location of these commands by typing

.. code-block:: bash
	
	which thunder

If you type ``thunder`` into the terminal it will start an interactive session in the Python shell.

If you want to upgrade to the latest version, we recommend first uninstalling using 

.. code-block:: bash
	
	pip uninstall thunder-python

And then install again to get the latest version.

Dependencies 
~~~~~~~~~~~~
Along with Spark, thunder depends on these Python libraries (by installing using ``pip`` these will be added automatically).

`numpy <http://www.numpy.org/>`__, `scipy <http://www.scipy.org/>`__, `scikit-learn <http://scikit-learn.org/stable/>`__, `matplotlib <matplotlib.sourceforge.net>`__

We recommend using the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_, which includes these dependencies (and many other useful packages). Especially if you aren't already using Python for scientific computing, it's a great way to start. 

iPython
~~~~~~~
Spark and thunder work well alongside iPython and the iPython notebook. To use iPython, just set an environmental variable

.. code-block:: bash

	export IPYTHON=1

When you type ``thunder`` it will start in iPython. If you additionally set

.. code-block:: bash

	export IPYTHON_OPTS="notebook --pylab inline"

it will use the iPython notebook. As before, you should add these lines to your bash profile.


