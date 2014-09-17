.. _install_ec2_ref:

Running on EC2
==============
Running Spark and Thunder on Amazon's EC2 is an easy way to quickly leverage the computational power of a large cluster. The following instructions assume you've already installed Thunder as described in :ref:`install_local_ref`.

Setting up an Amazon account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(You might be able to skip this step if you are already using EC2.) Go to `AWS <http://aws.amazon.com/>`_ to sign up for an account. Once you have created an account, go to `Identity and Access Management <https://console.aws.amazon.com/iam/#users>`_, select "Users" on the left, and click "Create new users" at the top. Follow the instructions to create a user for yourself. After you create a user, a window will pop up letting you "Show User Security Credentials". Click this to see your access key ID and secret access key (long strings of characters/numbers). Write these down, or click download to save them to a file, and close the window. While you have these handy, add the following two lines to your ``bash_profile``.

.. code-block:: bash

	export AWS_ACCESS_KEY_ID=<your-access-key-id>
	export AWS_SECRET_ACCESS_KEY=<your-secret-access-key>

Open a new terminal window so these changes take effect. Back on the AWS site, click the checkbox next to your user name. In the window that appears at the bottom of the page, select "Permissions > Attach user policy", select Administrator Access policy, and click Apply Policy. If you opened a brand new account it may take a couple hours for Amazon to verify it, so wait before proceeding.


Create your key pair
~~~~~~~~~~~~~~~~~~~~~~~
Go to the `Key Pair section on the EC2 console <https://console.aws.amazon.com/ec2/#KeyPairs:>`_. Click on the region name in the upper right (it will probably say Oregon) and select US East (N Virginia). Then click Create Key Pair at the top. Give it any name you want; we'll assume it's ``mykey``. You'll download a file called ``mykey.pem``; we'll assume you put it in your home folder. Set the correct permissions on this file by typing

.. code-block:: bash

	chmod 600 ~/mykey.pem

Launch a cluster
~~~~~~~~~~~~~~~~
Type the following command into the terminal

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem -s <number-of-nodes> launch <cluster-name>

where ``mykey`` is the key-pair that we just created, ``<number-of-nodes>`` is the number of nodes in your cluster (try 2 just for testing), and ``<cluster-name>`` is just an identifier and can be anything. This command will take a few minutes. You should see lots of activity in the terminal. If the cluster doesn't start, which sometimes happens because the instances are slow to boot up (it will repeatedly report the error: ``ssh: connect to host... Connection refused``), wait for it to end then resume:

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem launch <cluster-name> --resume

After it finishes you should see the message ``Cluster successfully launched!`` followed by a link to a URL. Go to this URL to see the Spark Web UI for your new cluster.

Login and start
~~~~~~~~~~~~~~~
To login just type

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem login <cluster-name>

Once logged in, start the Spark shell in iPython:

.. code-block:: bash

	thunder

Remember to ``exit`` the cluster and shut it down when you are done!

.. code-block:: bash
	
	thunder-ec2 -k mykey -i ~/mykey.pem destroy <cluster-name>

Use the iPython notebook
~~~~~~~~~~~~~~~~~~~~~~~~
The iPython notebook is an especially useful way to do analyses interactively and look at results.

To setup the iPython notebook on EC2, you need to do one manual port configuration on the AWS console website. Go to the `EC2 dashboard <https://console.aws.amazon.com/ec2/v2/home>`_, click on "Security groups" in the list on the left, and find the name of your cluster in the list, and click on the entry "<cluster-name>-master". So if you called your cluster "test", look for "test-master". After selecting it, in the panel below, click the "Inbound" tab, click "Edit", click "Add rule", then type 8888 in the port range, and select "Anywhere" under source, then click "Save". 

The rest is easy. Just login to your cluster

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem login <cluster-name>

and type:

.. code-block:: bash

	setup-notebook

This will run a script that configures an iPython notebook server. During the script you will be asked to enter a password. Rememember what you give, as we'll need it again soon. At the end of the configuration you'll see the message ``iPython notebook successfully set up!`` followed by a link to a URL. If you now type:

.. code-block:: bash

	source /root/.bash_profile
	thunder

and go to the URL from the previous step in a web browser, it should ask for the password we gave (if you get a message about SSL security, just click proceed). You are now running an iPython notebook server! Click ``New Notebook`` to start a session.







