#!/usr/bin/env python
"""
Wrapper for the Spark EC2 launch script that additionally
installs Anaconda, Thunder, and its dependencies, and optionally
loads example data sets
"""
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from Spark's spark_ec2.py under the terms of the ASF 2.0 license.

import sys
import os
import random
import subprocess
import time
from termcolor import colored
from distutils.version import LooseVersion
from sys import stderr
from optparse import OptionParser


from thunder import __version__ as THUNDER_VERSION

spark = os.getenv('SPARK_HOME')
if spark is None or spark == '':
    raise Exception('must assign the environmental variable SPARK_HOME with the location of Spark')

if os.getenv('PYTHONPATH') is None:
    os.environ['PYTHONPATH'] = ""

sys.path.insert(0,spark+"/ec2/")
sys.path.insert(0,spark+"/ec2/third_party/boto-2.4.1.zip/boto-2.4.1")
from spark_ec2 import launch_cluster, get_existing_cluster, stringify_command,\
    deploy_files, get_spark_ami, ssh_read, ssh_write
try:
    from spark_ec2 import wait_for_cluster
except ImportError:
    from spark_ec2 import wait_for_cluster_state

from boto import ec2

MINIMUM_SPARK_VERSION = "1.1.0"

EXTRA_SSH_OPTS = ['-o', 'UserKnownHostsFile=/dev/null',
                  '-o', 'CheckHostIP=no',
                  '-o', 'LogLevel=quiet']


def print_status(msg):
    print("    [" + msg + "]")


def print_success(msg="success"):
    print("    [" + colored(msg, 'green') + "]")


def print_error(msg="failed"):
    print("    [" + colored(msg, 'red') + "]")


class quiet(object):
    """ Minmize stdout and stderr from external processes """
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def get_s3_keys():
    """ Get user S3 keys from environmental variables """
    if os.getenv('S3_AWS_ACCESS_KEY_ID') is not None:
        s3_access_key = os.getenv("S3_AWS_ACCESS_KEY_ID")
    else:
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    if os.getenv('S3_AWS_SECRET_ACCESS_KEY') is not None:
        s3_secret_key = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
    else:
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    return s3_access_key, s3_secret_key


def get_default_thunder_version():
    """
    Returns 'HEAD' (current state of thunder master branch) if thunder is a dev version, otherwise
    return the current thunder version.
    """
    if ".dev" in THUNDER_VERSION:
        return 'HEAD'
    return THUNDER_VERSION


def get_spark_version_string(default_version):
    """
    Parses out the Spark version string from $SPARK_HOME/RELEASE, if present, or from pom.xml if not

    Returns version string from either of the above sources, or default_version if nothing else works
    """
    SPARK_HOME = os.getenv("SPARK_HOME")
    if SPARK_HOME is None:
        raise Exception('must assign the environmental variable SPARK_HOME with the location of Spark')
    if os.path.isfile(os.path.join(SPARK_HOME, "RELEASE")):
        with open(os.path.join(SPARK_HOME, "RELEASE")) as f:
            line = f.read()
        # some nasty ad-hoc parsing here. we expect a string of the form
        # "Spark VERSION built for hadoop HADOOP_VERSION"
        # where VERSION is a dotted version string.
        # for now, simply check that there are at least two tokens and the second token contains a period.
        tokens = line.split()
        if len(tokens) >= 2 and '.' in tokens[1]:
            return tokens[1]
    # if we get to this point, we've failed to parse out a version string from the RELEASE file. note that
    # there will not be a RELEASE file for versions of Spark built from source. in this case we'll try to
    # get it out from the pom file.
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(os.path.join(SPARK_HOME, "pom.xml"))
        version_elt = root.find("./{http://maven.apache.org/POM/4.0.0}version")
        if version_elt is not None:
            return version_elt.text
    except IOError:
        # no pom file; fall through and return default
        pass
    return default_version

SPARK_VERSIONS_TO_HASHES = {
    '1.2.0rc2': "a428c446e23e"
}


def setup_spark_cluster(master, opts):
    ssh(master, opts, "chmod u+x spark-ec2/setup.sh")
    ssh(master, opts, "spark-ec2/setup.sh")


def remap_spark_version_to_hash(user_version_string):
    """
    Replaces a user-specified Spark version string with a github hash if needed.

    Used to allow clusters to be deployed with Spark release candidates.
    """
    return SPARK_VERSIONS_TO_HASHES.get(user_version_string, user_version_string)


def install_anaconda(master, opts):
    """ Install Anaconda on a Spark EC2 cluster """

    # download anaconda
    print_status("Downloading Anaconda")
    ssh(master, opts, "wget http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/"
                      "Anaconda-2.1.0-Linux-x86_64.sh")
    print_success()

    # setup anaconda
    print_status("Installing Anaconda")
    ssh(master, opts, "rm -rf /root/anaconda && bash Anaconda-2.1.0-Linux-x86_64.sh -b "
                      "&& rm Anaconda-2.1.0-Linux-x86_64.sh")
    ssh(master, opts, "echo 'export PATH=/root/anaconda/bin:$PATH:/root/spark/bin' >> /root/.bash_profile")
    ssh(master, opts, "pssh -h /root/spark-ec2/slaves 'echo 'export "
                      "PATH=/root/anaconda/bin:$PATH:/root/spark/bin' >> /root/.bash_profile'")
    print_success()

    # update core libraries
    print_status("Updating Anaconda libraries")
    ssh(master, opts, "/root/anaconda/bin/conda update --yes numpy scipy ipython")
    ssh(master, opts, "/root/anaconda/bin/conda install --yes jsonschema pillow seaborn scikit-learn")
    print_success()

    # add mistune (for notebook conversions)
    ssh(master, opts, "source ~/.bash_profile && pip install mistune")

    # copy to slaves
    print_status("Copying Anaconda to workers")
    ssh(master, opts, "/root/spark-ec2/copy-dir /root/anaconda")
    print_success()


def install_thunder(master, opts):
    """ Install Thunder and dependencies on a Spark EC2 cluster """
    print_status("Installing Thunder")

    # download thunder
    ssh(master, opts, "rm -rf thunder && git clone https://github.com/freeman-lab/thunder.git")
    if opts.thunder_version.lower() != "head":
        tagOrHash = opts.thunder_version
        if '.' in tagOrHash and not (tagOrHash.startswith('v')):
            # we have something that looks like a version number. prepend 'v' to get a valid tag id.
            tagOrHash = 'v' + tagOrHash
        ssh(master, opts, "cd thunder && git checkout %s" % tagOrHash)

    # copy local data examples to all workers
    ssh(master, opts, "yum install -y pssh")
    ssh(master, opts, "pssh -h /root/spark-ec2/slaves mkdir -p /root/thunder/python/thunder/utils/data/")
    ssh(master, opts, "~/spark-ec2/copy-dir /root/thunder/python/thunder/utils/data/")

    # install requirements
    ssh(master, opts, "source ~/.bash_profile && pip install -r /root/thunder/python/requirements.txt")
    ssh(master, opts, "pssh -h /root/spark-ec2/slaves 'source ~/.bash_profile && pip install zope.cachedescriptors'")

    # set environmental variables
    ssh(master, opts, "echo 'export SPARK_HOME=/root/spark' >> /root/.bash_profile")
    ssh(master, opts, "echo 'export PYTHONPATH=/root/thunder/python' >> /root/.bash_profile")
    ssh(master, opts, "echo 'export IPYTHON=1' >> /root/.bash_profile")

    # build thunder
    ssh(master, opts, "chmod u+x thunder/python/bin/build")
    ssh(master, opts, "source ~/.bash_profile && thunder/python/bin/build")
    ssh(master, opts, "echo 'export PATH=/root/thunder/python/bin:$PATH' >> /root/.bash_profile")

    # add AWS credentials to ~/.boto
    access, secret = get_s3_keys()
    credentialstring = "[Credentials]\naws_access_key_id = ACCESS\naws_secret_access_key = SECRET\n"
    credentialsfilled = credentialstring.replace('ACCESS', access).replace('SECRET', secret)
    ssh(master, opts, "printf '"+credentialsfilled+"' > /root/.boto")
    ssh(master, opts, "printf '[s3]\ncalling_format = boto.s3.connection.OrdinaryCallingFormat' >> /root/.boto")
    ssh(master, opts, "pscp.pssh -h /root/spark-ec2/slaves /root/.boto /root/.boto")

    print_success()


def configure_spark(master, opts):
    """ Configure Spark with useful settings for running Thunder """
    print_status("Configuring Spark for Thunder")

    # customize spark configuration parameters
    ssh(master, opts, "echo 'spark.akka.frameSize=2047' >> /root/spark/conf/spark-defaults.conf")
    ssh(master, opts, "echo 'spark.kryoserializer.buffer.max.mb=1024' >> /root/spark/conf/spark-defaults.conf")
    ssh(master, opts, "echo 'spark.driver.maxResultSize=0' >> /root/spark/conf/spark-defaults.conf")
    ssh(master, opts, "echo 'export SPARK_DRIVER_MEMORY=20g' >> /root/spark/conf/spark-env.sh")
    ssh(master, opts, "sed 's/log4j.rootCategory=INFO/log4j.rootCategory=ERROR/g' "
                      "/root/spark/conf/log4j.properties.template > /root/spark/conf/log4j.properties")

    # point spark to the anaconda python
    ssh(master, opts, "echo 'export PYSPARK_DRIVER_PYTHON=/root/anaconda/bin/python' >> "
                      "/root/spark/conf/spark-env.sh")
    ssh(master, opts, "echo 'export PYSPARK_PYTHON=/root/anaconda/bin/python' >> "
                      "/root/spark/conf/spark-env.sh")
    ssh(master, opts, "/root/spark-ec2/copy-dir /root/spark/conf")

    # add AWS credentials to core-site.xml
    configstring = "<property><name>fs.s3n.awsAccessKeyId</name><value>ACCESS</value></property><property>" \
                   "<name>fs.s3n.awsSecretAccessKey</name><value>SECRET</value></property>"
    access, secret = get_s3_keys()
    filled = configstring.replace('ACCESS', access).replace('SECRET', secret)
    ssh(master, opts, "sed -i'f' 's,.*</configuration>.*,"+filled+"&,' /root/ephemeral-hdfs/conf/core-site.xml")
    ssh(master, opts, "sed -i'f' 's,.*</configuration>.*,"+filled+"&,' /root/spark/conf/core-site.xml")

    # configure requester pays
    ssh(master, opts, "touch /root/spark/conf/jets3t.properties")
    ssh(master, opts, "echo 'httpclient.requester-pays-buckets-enabled = true' >> /root/spark/conf/jets3t.properties")
    ssh(master, opts, "~/spark-ec2/copy-dir /root/spark/conf")

    print_success()


# This is a customized version of the spark_ec2 ssh() function that
# adds additional options to squash ssh host key checking errors that
# occur when the ip addresses of your ec2 nodes change when you
# start/stop a cluster.  Lame to have to copy all this code over, but
# this seemed the simplest way to add this necessary functionality.
def ssh_args(opts):
    parts = ['-o', 'StrictHostKeyChecking=no'] + EXTRA_SSH_OPTS
    # Never store EC2 IPs in known hosts...
    if opts.identity_file is not None:
        parts += ['-i', opts.identity_file]
    return parts


def ssh_command(opts):
    return ['ssh'] + ssh_args(opts)


def ssh(host, opts, command):
    tries = 0
    cmd = ssh_command(opts) + ['-t', '-t', '%s@%s' % (opts.user, host), stringify_command(command)]
    while True:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout = process.communicate()[0]
        code = process.returncode
        if code != 0:
            if tries > 2:
                print_error("SSH failure, returning error")
                raise Exception(stdout)
            else:
                time.sleep(3)
                tries += 1
        else:
            return


def setup_cluster(conn, master_nodes, slave_nodes, opts, deploy_ssh_key):
    """
    Modified version of the setup_cluster function (borrowed from spark-ec.py)
    in order to manually set the folder with the deploy code
    """
    master = master_nodes[0].public_dns_name
    if deploy_ssh_key:
        print_status("Generating cluster's SSH key on master")
        key_setup = """
      [ -f ~/.ssh/id_rsa ] ||
        (ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa &&
         cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys)
        """
        ssh(master, opts, key_setup)
        print_success()
        with quiet():
            dot_ssh_tar = ssh_read(master, opts, ['tar', 'c', '.ssh'])
        print_status("Transferring cluster's SSH key to slaves")
        with quiet():
            for slave in slave_nodes:
                ssh_write(slave.public_dns_name, opts, ['tar', 'x'], dot_ssh_tar)
        print_success()

    modules = ['spark', 'shark', 'ephemeral-hdfs', 'persistent-hdfs',
               'mapreduce', 'spark-standalone', 'tachyon']

    if opts.hadoop_major_version == "1":
        modules = filter(lambda x: x != "mapreduce", modules)

    if opts.ganglia:
        modules.append('ganglia')

    if spark_home_loose_version >= LooseVersion("1.3.0"):
        MESOS_SPARK_EC2_BRANCH = "branch-1.3"
        ssh(master, opts, "rm -rf spark-ec2 && git clone https://github.com/mesos/spark-ec2.git "
                          "-b {b}".format(b=MESOS_SPARK_EC2_BRANCH))
    else:
        ssh(master, opts, "rm -rf spark-ec2 && git clone https://github.com/mesos/spark-ec2.git "
                          "-b v4")

    print_status("Deploying files to master")
    deploy_folder = os.path.join(spark, "ec2", "deploy.generic")
    with quiet():
        deploy_files(conn, deploy_folder, opts, master_nodes, slave_nodes, modules)
    print_success()

    print_status("Installing Spark (may take several minutes)")
    setup_spark_cluster(master, opts)
    print_success()

def main():
    spark_home_version_string = get_spark_version_string(MINIMUM_SPARK_VERSION)
    spark_home_loose_version = LooseVersion(spark_home_version_string)

    parser = OptionParser(usage="thunder-ec2 [options] <action> <clustername>",  add_help_option=False)
    parser.add_option("-h", "--help", action="help", help="Show this help message and exit")
    parser.add_option("-k", "--key-pair", help="Key pair to use on instances")
    parser.add_option("-s", "--slaves", type="int", default=1, help="Number of slaves to launch (default: 1)")
    parser.add_option("-i", "--identity-file", help="SSH private key file to use for logging into instances")
    parser.add_option("-r", "--region", default="us-east-1", help="EC2 region zone to launch instances "
                                                                  "in (default: us-east-1)")
    parser.add_option("-t", "--instance-type", default="m3.2xlarge",
                      help="Type of instance to launch (default: m3.2xlarge)." +
                           " WARNING: must be 64-bit; small instances won't work")
    parser.add_option(
        "-m", "--master-instance-type", default="",
        help="Master instance type (leave empty for same as instance-type)")
    parser.add_option("-u", "--user", default="root", help="User name for cluster (default: root)")
    parser.add_option("-v", "--spark-version", default=spark_home_version_string,
                      help="Version of Spark to use: 'X.Y.Z' or a specific git hash. (default: %s)" %
                           spark_home_version_string)
    parser.add_option("--thunder-version", default=get_default_thunder_version(),
                      help="Version of Thunder to use: 'X.Y.Z', 'HEAD' (current state of master branch), " +
                           " or a specific git hash. (default: '%default')")

    if spark_home_loose_version >= LooseVersion("1.2.0"):
        parser.add_option(
            "-w", "--wait", type="int", default=160,
            help="DEPRECATED (no longer necessary for Spark >= 1.2.0) - Seconds to wait for nodes to start")
    else:
        parser.add_option("-w", "--wait", type="int", default=160,
                          help="Seconds to wait for nodes to start (default: 160)")
    parser.add_option("-z", "--zone", default="", help="Availability zone to launch instances in, or 'all' to spread "
                                                       "slaves across multiple (an additional $0.01/Gb for "
                                                       "bandwidth between zones applies)")
    parser.add_option(
        "--spark-git-repo",
        default="https://github.com/apache/spark",
        help="Github repo from which to checkout supplied commit hash")
    parser.add_option(
        "--hadoop-major-version", default="1",
        help="Major version of Hadoop (default: %default)")
    parser.add_option("--ssh-port-forwarding", default=None,
                      help="Set up ssh port forwarding when you login to the cluster.  " +
                           "This provides a convenient alternative to connecting to iPython " +
                           "notebook over an open port using SSL.  You must supply an argument " +
                           "of the form \"local_port:remote_port\".")
    parser.add_option(
        "--ebs-vol-size", metavar="SIZE", type="int", default=0,
        help="Size (in GB) of each EBS volume.")
    parser.add_option(
        "--ebs-vol-type", default="standard",
        help="EBS volume type (e.g. 'gp2', 'standard').")
    parser.add_option(
        "--ebs-vol-num", type="int", default=1,
        help="Number of EBS volumes to attach to each node as /vol[x]. " +
             "The volumes will be deleted when the instances terminate. " +
             "Only possible on EBS-backed AMIs. " +
             "EBS volumes are only attached if --ebs-vol-size > 0." +
             "Only support up to 8 EBS volumes.")
    parser.add_option(
        "--swap", metavar="SWAP", type="int", default=1024,
        help="Swap space to set up per node, in MB (default: %default)")
    parser.add_option("--spot-price", metavar="PRICE", type="float",
                      help="If specified, launch slaves as spot instances with the given " +
                           "maximum price (in dollars)")
    parser.add_option(
        "--ganglia", action="store_true", default=True,
        help="Setup Ganglia monitoring on cluster (default: %default). NOTE: " +
             "the Ganglia page will be publicly accessible")
    parser.add_option(
        "--no-ganglia", action="store_false", dest="ganglia",
        help="Disable Ganglia monitoring for the cluster")
    parser.add_option("--resume", default=False, action="store_true",
                      help="Resume installation on a previously launched cluster (for debugging)")
    parser.add_option(
        "--worker-instances", type="int", default=1,
        help="Number of instances per worker: variable SPARK_WORKER_INSTANCES (default: %default)")
    parser.add_option("--master-opts", type="string", default="",
                      help="Extra options to give to master through SPARK_MASTER_OPTS variable " +
                           "(e.g -Dspark.worker.timeout=180)")
    parser.add_option("--user-data", type="string", default="",
                      help="Path to a user-data file (most AMI's interpret this as an initialization script)")
    if spark_home_loose_version >= LooseVersion("1.2.0"):
        parser.add_option("--authorized-address", type="string", default="0.0.0.0/0",
                          help="Address to authorize on created security groups (default: %default)" +
                               " (only with Spark >= 1.2.0)")
        parser.add_option("--additional-security-group", type="string", default="",
                          help="Additional security group to place the machines in (only with Spark >= 1.2.0)")
        parser.add_option("--copy-aws-credentials", action="store_true", default=False,
                          help="Add AWS credentials to hadoop configuration to allow Spark to access S3" +
                               " (only with Spark >= 1.2.0)")
    if spark_home_loose_version >= LooseVersion("1.3.0"):
        parser.add_option("--subnet-id", default=None,
                          help="VPC subnet to launch instances in (only with Spark >= 1.3.0")
        parser.add_option("--vpc-id", default=None,
                          help="VPC to launch instances in (only with Spark >= 1.3.0)")
        parser.add_option("--placement-group", type="string", default=None,
                          help="Which placement group to try and launch instances into. Assumes placement "
                               "group is already created.")

    (opts, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
    (action, cluster_name) = args

    spark_version_string = opts.spark_version
    # check that requested spark version is <= the $SPARK_HOME version, or is a github hash
    if '.' in spark_version_string:
        # version string is dotted, not a hash
        spark_cluster_loose_version = LooseVersion(spark_version_string)
        if spark_cluster_loose_version > spark_home_loose_version:
            raise ValueError("Requested cluster Spark version '%s' is greater " % spark_version_string
                             + "than the local version of Spark in $SPARK_HOME, '%s'" % spark_home_version_string)
        if spark_cluster_loose_version < LooseVersion(MINIMUM_SPARK_VERSION):
            raise ValueError("Requested cluster Spark version '%s' is less " % spark_version_string
                             + "than the minimum version required for Thunder, '%s'" % MINIMUM_SPARK_VERSION)

    opts.ami = get_spark_ami(opts)  # "ami-3ecd0c56"\
    # get version string as github commit hash if needed (mainly to support Spark release candidates)
    opts.spark_version = remap_spark_version_to_hash(spark_version_string)

    # Launch a cluster, setting several options to defaults
    # (use spark-ec2.py included with Spark for more control)
    if action == "launch":
        try:
            conn = ec2.connect_to_region(opts.region)
        except Exception as e:
            print >> stderr, (e)
            sys.exit(1)

        if opts.zone == "":
            opts.zone = random.choice(conn.get_all_zones()).name

        if opts.resume:
            (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        else:
            (master_nodes, slave_nodes) = launch_cluster(conn, opts, cluster_name)

        try:
            wait_for_cluster(conn, opts.wait, master_nodes, slave_nodes)
        except NameError:
            if spark_home_loose_version >= LooseVersion("1.3.0"):
                wait_for_cluster_state(cluster_instances=(master_nodes + slave_nodes),
                                       cluster_state='ssh-ready', opts=opts, conn=conn)
            else:
                wait_for_cluster_state(cluster_instances=(master_nodes + slave_nodes),
                                       cluster_state='ssh-ready', opts=opts)
        print("")
        setup_cluster(conn, master_nodes, slave_nodes, opts, True)
        master = master_nodes[0].public_dns_name
        install_anaconda(master, opts)
        install_thunder(master, opts)
        configure_spark(master, opts)

        print("")
        print("Cluster successfully launched!")
        print("")
        print("Go to " + colored("http://%s:8080" % master, 'blue') + " to see the web UI for your cluster")
        if opts.ganglia:
            print("Go to " + colored("http://%s:5080/ganglia" % master, 'blue') + " to view ganglia monitor")
        print("")

    if action != "launch":
        conn = ec2.connect_to_region(opts.region)
        (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        master = master_nodes[0].public_dns_name

        # Login to the cluster
        if action == "login":
            print "Logging into master " + master + "..."
            proxy_opt = []

            # SSH tunnels are a convenient, zero-configuration
            # alternative to opening a port using the EC2 security
            # group settings and using iPython notebook over SSL.
            #
            # If the user has requested ssh port forwarding, we set
            # that up here.
            if opts.ssh_port_forwarding is not None:
                ssh_ports = opts.ssh_port_forwarding.split(":")
                if len(ssh_ports) != 2:
                    print "\nERROR: Could not parse arguments to \'--ssh-port-forwarding\'."
                    print "       Be sure you use the syntax \'local_port:remote_port\'"
                    sys.exit(1)
                print ("\nSSH port forwarding requested.  Remote port " + ssh_ports[1] +
                       " will be accessible at http://localhost:" + ssh_ports[0] + '\n')
                try:
                    subprocess.check_call(ssh_command(opts) + proxy_opt + EXTRA_SSH_OPTS +
                                          ['-L', ssh_ports[0] +
                                           ':127.0.0.1:' + ssh_ports[1],
                                           '-o', 'ExitOnForwardFailure=yes',
                                           '-t', '-t', "%s@%s" % (opts.user, master)])
                except subprocess.CalledProcessError:
                    print "\nERROR: Could not establish ssh connection with port forwarding."
                    print "       Check your Internet connection and make sure that the"
                    print "       ports you have requested are not already in use."
                    sys.exit(1)

            else:
                subprocess.check_call(ssh_command(opts) + proxy_opt + EXTRA_SSH_OPTS +
                                      ['-t', '-t',
                                       "%s@%s" % (opts.user, master)])

        elif action == "reboot-slaves":
            response = raw_input(
                "Are you sure you want to reboot the cluster " +
                cluster_name + " slaves?\n" +
                "Reboot cluster slaves " + cluster_name + " (y/N): ")
            if response == "y":
                (master_nodes, slave_nodes) = get_existing_cluster(
                    conn, opts, cluster_name, die_on_error=False)
                print "Rebooting slaves..."
                for inst in slave_nodes:
                    if inst.state not in ["shutting-down", "terminated"]:
                        print "Rebooting " + inst.id
                        inst.reboot()

        elif action == "get-master":
            print master_nodes[0].public_dns_name

        # Install thunder on the cluster
        elif action == "install":
            install_anaconda(master, opts)
            install_thunder(master, opts)
            configure_spark(master, opts)

        # Stop a running cluster.  Storage on EBS volumes is
        # preserved, so you can restart the cluster in the same state
        # (though you do pay a modest fee for EBS storage in the
        # meantime).
        elif action == "stop":
            response = raw_input(
                "Are you sure you want to stop the cluster " +
                cluster_name + "?\nDATA ON EPHEMERAL DISKS WILL BE LOST, " +
                "BUT THE CLUSTER WILL KEEP USING SPACE ON\n" +
                "AMAZON EBS IF IT IS EBS-BACKED!!\n" +
                "All data on spot-instance slaves will be lost.\n" +
                "Stop cluster " + cluster_name + " (y/N): ")
            if response == "y":
                (master_nodes, slave_nodes) = get_existing_cluster(
                    conn, opts, cluster_name, die_on_error=False)
                print "Stopping master..."
                for inst in master_nodes:
                    if inst.state not in ["shutting-down", "terminated"]:
                        inst.stop()
                print "Stopping slaves..."
                for inst in slave_nodes:
                    if inst.state not in ["shutting-down", "terminated"]:
                        if inst.spot_instance_request_id:
                            inst.terminate()
                        else:
                            inst.stop()

        # Restart a stopped cluster
        elif action == "start":
            print "Starting slaves..."
            for inst in slave_nodes:
                if inst.state not in ["shutting-down", "terminated"]:
                    inst.start()
            print "Starting master..."
            for inst in master_nodes:
                if inst.state not in ["shutting-down", "terminated"]:
                    inst.start()
            try:
                wait_for_cluster(conn, opts.wait, master_nodes, slave_nodes)
            except NameError:
                if spark_home_loose_version >= LooseVersion("1.3.0"):
                    wait_for_cluster_state(cluster_instances=(master_nodes + slave_nodes),
                                           cluster_state='ssh-ready', opts=opts, conn=conn)
                else:
                    wait_for_cluster_state(cluster_instances=(master_nodes + slave_nodes),
                                           cluster_state='ssh-ready', opts=opts)
            print("")
            setup_cluster(conn, master_nodes, slave_nodes, opts, False)
            master = master_nodes[0].public_dns_name
            configure_spark(master, opts)
            print("")
            print("Cluster successfully restarted!")
            print("Go to " + colored("http://%s:8080" % master, 'blue') + " to see the web UI for your cluster")
            print("")

        # Destroy the cluster
        elif action == "destroy":
            response = raw_input("Are you sure you want to destroy the cluster " + cluster_name +
                                 "?\nALL DATA ON ALL NODES WILL BE LOST!!\n" +
                                 "Destroy cluster " + cluster_name + " (y/N): ")
            if response == "y":
                (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name, die_on_error=False)
            print "Terminating master..."
            for inst in master_nodes:
                inst.terminate()
            print "Terminating slaves..."
            for inst in slave_nodes:
                inst.terminate()

        else:
            raise NotImplementedError("action: " + action + "not recognized")


if __name__ == "__main__":
    main()