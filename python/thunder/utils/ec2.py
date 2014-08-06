"""
Wrapper for the Spark EC2 launch script that additionally
installs Thunder and its dependencies, and optionally
loads an example data set
"""

from boto import ec2
import sys
import os
import time
import random
import subprocess
from sys import stderr
from optparse import OptionParser
from spark_ec2 import ssh, launch_cluster, get_existing_cluster, wait_for_cluster, deploy_files, setup_spark_cluster, \
    get_spark_ami, ssh_command, ssh_read, ssh_write


def get_s3_keys():
    """ Get user S3 keys from environmental variables"""
    if os.getenv('S3_AWS_ACCESS_KEY_ID') is not None:
        s3_access_key = os.getenv("S3_AWS_ACCESS_KEY_ID")
    else:
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    if os.getenv('S3_AWS_SECRET_ACCESS_KEY') is not None:
        s3_secret_key = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
    else:
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    return s3_access_key, s3_secret_key


def install_thunder(master, opts):
    """ Install Thunder and dependencies on a Spark EC2 cluster"""
    print "Installing Thunder on the cluster..."
    ssh(master, opts, "rm -rf thunder && git clone https://github.com/freeman-lab/thunder.git")
    ssh(master, opts, "chmod u+x thunder/python/bin/build")
    ssh(master, opts, "thunder/python/bin/build")
    ssh(master, opts, "source ~/.bash_profile && pip install mpld3")
    ssh(master, opts, "echo 'export SPARK_HOME=/root/spark' >> /root/.bash_profile")
    ssh(master, opts, "echo 'export PYTHONPATH=/root/thunder/python' >> /root/.bash_profile")
    ssh(master, opts, "echo 'export IPYTHON=1' >> /root/.bash_profile")
    ssh(master, opts, "echo 'export PATH=/root/thunder/python/bin:$PATH' >> /root/.bash_profile")
    print "\n\n"
    print "-------------------------------"
    print "Thunder successfully installed!"
    print "-------------------------------"
    print "\n"


def load_data(master, opts):
    """ 
    Load an example data set into a Spark EC2 cluster
    TODO: replace with URL once we've hosted public data
    """
    print "Transferring example data to the cluster..."
    ssh(master, opts, "/root/ephemeral-hdfs/bin/stop-all.sh")
    ssh(master, opts, "/root/ephemeral-hdfs/bin/start-all.sh")
    time.sleep(10)
    ssh(master, opts, "/root/ephemeral-hdfs/bin/hadoop distcp s3n://thunder.datasets/test/iris.txt hdfs:///data")
    print "\n\n"
    print "-------------------------------"
    print "Example data successfully loaded!"
    print "-------------------------------"
    print "\n"


def setup_cluster(conn, master_nodes, slave_nodes, opts, deploy_ssh_key):
    """Modified version of the setup_cluster function (borrowed from spark-ec.py)
    in order to manually set the folder with the deploy code"""
    master = master_nodes[0].public_dns_name
    if deploy_ssh_key:
        print "Generating cluster's SSH key on master..."
        key_setup = """
      [ -f ~/.ssh/id_rsa ] ||
        (ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa &&
         cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys)
        """
        ssh(master, opts, key_setup)
        dot_ssh_tar = ssh_read(master, opts, ['tar', 'c', '.ssh'])
        print "Transferring cluster's SSH key to slaves..."
        for slave in slave_nodes:
            print slave.public_dns_name
            ssh_write(slave.public_dns_name, opts, ['tar', 'x'], dot_ssh_tar)

    modules = ['spark', 'shark', 'ephemeral-hdfs', 'persistent-hdfs',
             'mapreduce', 'spark-standalone', 'tachyon']

    if opts.hadoop_major_version == "1":
        modules = filter(lambda x: x != "mapreduce", modules)

    if opts.ganglia:
        modules.append('ganglia')

    ssh(master, opts, "rm -rf spark-ec2 && git clone https://github.com/mesos/spark-ec2.git -b v3")

    print "Deploying files to master..."
    deploy_folder = os.path.join(os.environ['SPARK_HOME'], "ec2", "deploy.generic")
    deploy_files(conn, deploy_folder, opts, master_nodes, slave_nodes, modules)

    print "Running setup on master..."
    setup_spark_cluster(master, opts)
    print "Done!"


if __name__ == "__main__":
    parser = OptionParser(usage="thunder-ec2 [options] <action> <clustername>",  add_help_option=False)
    parser.add_option("-h", "--help", action="help", help="Show this help message and exit")
    parser.add_option("-k", "--key-pair", help="Key pair to use on instances")
    parser.add_option("-s", "--slaves", type="int", default=1, help="Number of slaves to launch (default: 1)")
    parser.add_option("-i", "--identity-file", help="SSH private key file to use for logging into instances")
    parser.add_option("-r", "--region", default="us-east-1", help="EC2 region zone to launch instances "
                                                                  "in (default: us-east-1)")
    parser.add_option("-t", "--instance-type", default="m1.large", help="Type of instance to launch (default: m1.large)."
                                                                        " WARNING: must be 64-bit; small instances "
                                                                        "won't work")
    parser.add_option("-u", "--user", default="root", help="User name for cluster (default: root)")
    parser.add_option("-z", "--zone", default="", help="Availability zone to launch instances in, or 'all' to spread "
                                                       "slaves across multiple (an additional $0.01/Gb for "
                                                       "bandwidth between zones applies)")
    parser.add_option("--resume", default=False, action="store_true", help="Resume installation on a previously "
                                                        "launched cluster (for debugging)")

    (opts, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
    (action, cluster_name) = args

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

        opts.ami = "ami-3ecd0c56"
        opts.ebs_vol_size = 0
        opts.spot_price = None
        opts.master_instance_type = ""
        opts.wait = 160
        opts.hadoop_major_version = "1"
        opts.ganglia = True
        opts.spark_version = "1.0.1"
        opts.swap = 1024
        opts.worker_instances = 1
        opts.master_opts = ""

        if opts.resume:
            (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        else:
            (master_nodes, slave_nodes) = launch_cluster(conn, opts, cluster_name)

        wait_for_cluster(conn, opts.wait, master_nodes, slave_nodes)
        setup_cluster(conn, master_nodes, slave_nodes, opts, True)
        master = master_nodes[0].public_dns_name
        install_thunder(master, opts)
        print "\n\n"
        print "-------------------------------"
        print "Cluster successfully launched!"
        print "Go to http://%s:8080 to see the web UI for your cluster" % master
        print "-------------------------------"
        print "\n"

    if action != "launch":
        conn = ec2.connect_to_region(opts.region)
        (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        master = master_nodes[0].public_dns_name

        # Login to the cluster
        if action == "login":
            print "Logging into master " + master + "..."
            proxy_opt = []
            subprocess.check_call(ssh_command(opts) + proxy_opt + ['-t', '-t', "%s@%s" % (opts.user, master)])

        # Install thunder on the cluster
        elif action == "install":
            install_thunder(master, opts)

        # Load example data into the cluster
        elif action == "loaddata":
            load_data(master, opts)

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


