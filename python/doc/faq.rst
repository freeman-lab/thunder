.. _contributing:

.. currentmodule:: thunder

F.A.Q.
======

Basic usage
~~~~~~~~~~~

*When should I cache my data?*
	
	Any data object can be cached into RAM by calling ``data.cache()`` on it. Caching is a powerful capability of Spark, and can greatly speed up subsequent queries or operations on the data. However, you can only cache as much data as your cluster has RAM (and the actual RAM available to Spark for caching is usually approximately half of the total installed RAM of the cluster). It often makes sense to cache intermediate results, for example, after filtering or preprocessing. You should rarely cache data you only plan to use once; conversely, you should almost always cache data that will be subjected to iterative algorithms (such as clustering).

*I performed an operation and it happened instantaneously, is it really that fast?*

	Probably not. Many distributed operations in Thunder and Spark are lazy, which means they will not be executed until the necessary output requires them to be. For example, if you call ``images.medianFilter().sum()`` on an ``Images`` object, the first filtering operation is lazy, and is computed only because you asked for the sum. If you had just called ``imgs.medianFilter()`` it would appear to have happened immediately, but nothing would have actually been computed (yet).

*What's the fastest way to inspect my data?*

	Data objects are collections of records, and you can call `.first()` on any data object to display the first record of the data set. This will return a tuple, where the first element is the key and the second element is the value.

*Can I index directly into an Images or Series object?*

	Sort of! You can use bracket notation to directly extract a record with a particular key. For example, if `imgs` is an `Images` object, you can use `imgs[0]` to get the record with a key of 0, and if `series` is a `Series` object you can use `series[50, 40, 1]` to get the record with key `(50, 40, 1)`. You can also grab a range of records using slice notation, as in `series[0:10, 0:10, 0:1]`. These operations are syntactic sugar for using the `filter` operation. Also note that the speed of these operations will depend on the size of the dataset and whether or not it's been cached. We hope to support even faster direct indexing in the future.

*I’m routinely converting my Images data to Series and only ever working with the Series data, can I avoid this conversion step?*

	Yes, you can! You just have to save your Images data to binary Series data. This requires storing an extra copy of the data (bad), but will make it faster to load data in the future (good). The ``ThunderContext`` provides a method for this conversion, ``convertImagesToSeries``, which takes as arguments an input path for the Images data and an output path (typically either a networked file system volume or AWS S3 bucket) to which to write the Series output. The Series data can subsequently be loaded via the ``tsc.loadSeries`` method. Alternatively, one could load the image data by ``tsc.loadImages``, and then save it with the ``saveAsBinarySeries`` method. 

*Is there sample data I can try analyses on?*
	
	Yes! For local testing on tiny data sets, you can load data using `tsc.loadExample`, as in `data = tsc.loadExample('fish-series')`. These are tiny toy data sets, meant only for understanding the API and not for any meaningful analysis. If you are running on an AWS cluster, you can load example data sets from S3 using the `tsc.loadExampleS3` method. This includes both light-sheet imaging and two-photon imaging data, made availiable through the CodeNeuro `data portal <http://datasets.codeneuro.org>`_.

*How should I visualize the results of my analysis?*
	
	We recommend running Thunder in an iPython notebook. For simple plots, you can use ``matplotlib``, along with the excellent library ``seaborn``, to look at the output of many analyses directly in the notebook. We are currently developing a separate library, `Lightning <https://github.com/lightning-viz/lightning>`_, for interactive visualization, with many visualizations tailored to render the output of Thunder analyses. It can be used for standalone visualizations, or within the iPython notebook. 

*I’m running Thunder locally and it's not super fast, why is that?*
	
	Thunder is designed to run on a cluster. One nice feature is that you can run entirely locally, which is a great way to debug. For small data sets that can be loaded entirely on a single machine, simple operations (e.g. computing a mean) may be slower in Thunder than when using purely local operations. But more computationally-intensive operations (e.g. clustering, complex regression) will still benefit greatly from even a small cluster. We are investigating strategies for speeding up purely local workflows.

*I just did a series.pack() and the array I get back is all screwed up!*
	
	Try specifying ``sorting=True`` in the ``pack()`` call. The most likely problem is the ordering of your data. ``Series.pack()`` assumes by default that its keys are in "series-sorted" order, with the ``x`` dimension changing most rapidly, followed by ``y``, and finally ``z``. If you create series data within Thunder (e.g. from an ``Images`` object), it should be in this order already. But if you created the data yourself, or under some other conditions, it may have the wrong order and need sorting. 

*When trying to work with binary Series data, I get an error similar to the following: "java.lang.IncompatibleClassChangeError: Found interface org.apache.hadoop.mapreduce.JobContext, but class was expected"*
	
	This error indicates that the Spark Hadoop version that Thunder was compiled against doesn’t match the version that is actually installed on the cluster. The jar files distributed with Thunder by default are compiled against Hadoop 1.0.4, which is what you will get with any of the “Pre-built for Hadoop 1.x” precompiled Spark distributions available from `here <https://spark.apache.org/downloads.html>`_. If Hadoop 2.x jars are found on the classpath at runtime rather than Hadoop 1.x jars, then you will see this error anytime the Thunder jar is used. The simplest workaround would be to make sure that ``$SPARK_HOME`` refers to a version of Spark compiled against Hadoop 1.x. If you prefer to use a version of Spark compiled for Hadoop 2.x, then at present this requires Thunder to be rebuilt from source. (See instructions elsewhere).

Configuration and installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*My university / research institute has a private cluster, can I install Spark on it?*

	Yes. We are preparing a how-to that explains how to do this.

	Here are some instructions to set up thunder on a standalone (SMP) server. Tested on Mac OS X and Debian 7 (wheezy).

	* Install `Spark <https://spark.apache.org/downloads.html>`_. Select Spark |ge| 1.2.0, Pre-built for Hadoop 1.X (not sure how much the Hadoop bit matters - we won't use Hadoop).

	  .. |ge| unicode:: U+2265

	* Edit ``$SPARK_HOME/conf/spark-defaults.conf`` and add the following lines::

	    spark.driver.memory              20g
            spark.driver.maxResultSize       20g

	  You may have to increase these values in case you use larger data sets.

	* Thunder users will have to increase their open file limit before they launch thunder. Add the following line to the bottom of ``/etc/security/limits.conf``::

	    *               hard    nofile             1048576

	  All thunder users should then add the following to their ``.bashrc`` (or ``.profile`` on OS X)::

	    ulimit -Sn unlimited

	* The following lines in your ``.bashrc`` (or ``.profile`` on OS X) will set up ``SPARK_HOME`` and ``SPARK_LOCAL_DIRS`` during login. It makes sense to use a separate tmp directory for spark (``SPARK_LOCAL_DIRS``) to make it easier to clean up the mess it leaves behind from time to time. Make sure to ``mkdir ${HOME}/spark-tmp`` before you start spark for the first time.::

	    export SPARK_HOME=/usr/local/spark
	    export SPARK_LOCAL_DIRS=${HOME}/spark-tmp

*How many cores / cluster nodes / total amount of RAM do I need?*

	In general, fewer nodes with more cores and RAM is better than more, under-powered nodes, because it minimizes network communication (which is a limiting factor in some, though not all, workflows). For RAM, a good rule of thumb is to determine the size of your data, or at least the portion you want to cache, and then use a cluster with at least twice that much RAM in total. For numbers of nodes and cores, you want at least enough cores so that there are approximately 2-3 partitions of data per core, and partitions are usually ~50MB. So as an example, for an 100GB data set with 2000 partitions, you would do well for memory and computing with 10 nodes each having 32 cores and 64GB RAM, for a total of 320 cores and 640GB RAM.

*What configuration parameters should I use?*

	Spark has many `configurable parameters <http://spark.apache.org/docs/latest/configuration.html>`_. The following settings have generally proven useful when running Thunder, and are automatically set by Thunder's EC2 scripts:

	.. code-block:: python 

		spark.akka.frameSize=10000
		spark.kryoserializer.buffer.max.mb=1024
		spark.driver.maxResultSize=0


Using Thunder on Amazon's EC2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*How expensive is S3 and EC2?*
	
	For current on-demand pricing by instance type, see the `pricing <http://aws.amazon.com/ec2/pricing/>`_ guide from Amazon. You can also view the `instance types <http://aws.amazon.com/ec2/instance-types/>`_, in particular the matrix at the bottom of that page. This `web app <http://www.ec2instances.info/>`_ is also a useful resource. For details on storage charges, see `here <http://aws.amazon.com/s3/pricing/>`_.

*Where should I store my data?*
	
	Options are S3 and glacier.

*I want to share my data, what settings should I use?*
	
	You can make your data fully public by following these `instructions <https://ariejan.net/2010/12/24/public-readable-amazon-s3-bucket-policy/>`_. This means that anyone can load your data from within an EC2 cluser, or download it. Loading from within an EC2 is free (subject to being in the same region), but downloading it incurs transfer costs, for which you will be responsible. An alternative is to set your bucket to be `Requester Pays <http://docs.aws.amazon.com/AmazonS3/latest/dev/RequesterPaysBuckets.html>`_. This ensures that the data are accessible from within EC2 but not available for direct download without authentication.

*What EC2 instance types should I use?*
	
	Thunder workflows may be bottlenecked either by I/O or by computational limitations, depending on the nature of the workflow. Smaller EC2 instances have both lesser compute capacity and more modest network bandwidth between cluster nodes. Provisioning the same total number of cores spread across a cluster of smaller instances also increases the amount of network traffic required, relative to a cluster composed of fewer but larger instances. Finally, EC2 pricing is generally roughly equivalent for the same total amount of memory and CPUs across a cluster, no matter whether these resources are concentrated in a few large instances or spread across many smaller ones. Overall, this argues for using fewer, larger instances. With Spark, I/O limits can often be effectively addressed by caching in memory (after an initial read from disk or S3), suggesting that high-memory (“r”-type) instances may be a good choice for I/O-bound workflows. Conversely, compute-optimized (“c”-type) instances are likely most cost effective for computationally-intensive workflows. Both these instance types also have 10GB networking available for the largest instances. The thunder-ec2 script currently defaults to m3.2xlarge instances, which represent a balance between memory and compute capacity. 

*I’m concerned about the costs of running a large EC2 cluster. How can I save money?*

	Consider using `spot instances <http://aws.amazon.com/ec2/purchasing-options/spot-instances/>`_. These allow you to set a maximum hourly price that you would be willing to pay for an instance. You will be charged at the spot instance rate for your particular availability zone and instance type, which reflects the current available capacity at the AWS data centers and which will typically be much less than the standard on-demand rate. The downside is that if the spot rate ever exceeds your bid, then you will lose the instances; thus, spot instances are not recommended for last-minute analyses before an abstract submission deadline or similar. However, in practice, spot instances can be very long-lived, and represent a practical and affordable alternative for most data analysis tasks. 

*I'm getting a message saying that I've requested too many instances*

	By default, AWS will limit the maximum number of instances or spot instances of a given type that can be running under a single account. The high-memory (“r”-type) nodes in particular have relatively low spot instance limits, see `here <http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html>`_. 

*I get an error saying "The key pair 'mykey.pem' does not exist"*

	This is probably because you have the ``-k`` and ``-i`` arguments flipped in your call to ``thunder-ec2``. The command should look like:

	.. code-block:: python

		thunder-ec2 -k mykey -i mykey.pem launch clustername

*I see repeated errors when starting a cluster on Amazon EC2, what should I do?*

	You will most likely see the following error:

	.. code-block:: python 

		ssh: connect to host ec2-54-197-68-74.compute-1.amazonaws.com port 22: Connection refused
		Error 255 while executing remote command, retrying after 30 seconds

	This probably indicates that for whatever reason, one or more of the EC2 nodes has been slow to start. Oftentimes this will resolve itself; after waiting ~2-3 minutes, try resuming the cluster launch, by adding the --resume flag to the same thunder-ec2 command line as previously. If that fails a couple times, you can simply terminate the existing cluster and try again. This most likely represents a transient AWS issue, rather than a problem with Thunder or Spark per se. This issue also appears to be greatly reduced as of Spark 1.2, which will automatically wait for all instances to be in an “ssh-ready” state before attempting to connect. If you encounter this problem repeatedly, consider updating the installation of Spark on the machine you use to launch the EC2 cluster to Spark 1.2.0 or later. 






