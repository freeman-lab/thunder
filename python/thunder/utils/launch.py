""" Functions used in the Thunder launch scripts in python/bin """

import glob
import os
import subprocess

import thunder


def getSparkHome():
    sparkhome = os.getenv("SPARK_HOME")
    if sparkhome is None:
        raise Exception("The environment variable SPARK_HOME must be set to the Spark installation directory")
    if not os.path.exists(sparkhome):
        raise Exception("No Spark installation at %s, check that SPARK_HOME is correct" % sparkhome)
    return sparkhome


def getMasterURI(optsDict):
    if "master" in optsDict:
        master = optsDict["master"]
    elif "MASTER" in os.environ:
        master = os.getenv("MASTER")
    elif isEC2():
        master = getEC2Master()
    else:
        master = "local"
    return master


def isEC2():
    return os.path.isfile('/root/spark-ec2/cluster-url')


def getEC2Master():
    """
    Returns the cluster master URI, read from /root/spark-ec2/cluster-url.

    This file is expected to exist on EC2 clusters. An exception will be thrown if the file is missing.
    """
    with open('/root/spark-ec2/cluster-url', 'r') as f:
        master = f.read().strip()
    return master


def findThunderEgg():
    thunderdir = os.path.dirname(os.path.realpath(thunder.__file__))
    egg = glob.glob(os.path.join(thunderdir, 'lib', 'thunder_python-'+str(thunder.__version__)+'*.egg'))
    if len(egg) == 1:
        return egg[0]
    else:
        return None


def cleanThunderEgg():
    calldir = os.path.dirname(os.path.realpath(__file__))
    egg = 'thunder_python-' + str(thunder.__version__) + '*.egg'
    existing = glob.glob(os.path.join(calldir, '..', '..', 'thunder/lib/', egg))
    for f in existing:
        os.remove(f)


def buildThunderEgg():
    import shutil
    cleanThunderEgg()
    calldir = os.path.dirname(os.path.realpath(__file__))
    pythondir = os.path.join(calldir, '..', '..')
    subprocess.check_call(["python", 'setup.py', 'clean', 'bdist_egg'], cwd=pythondir)
    egg = 'thunder_python-' + str(thunder.__version__) + '*.egg'
    src = glob.glob(os.path.join(calldir, '..', '..', 'dist', egg))
    target = os.path.join(calldir, '..', '..', 'thunder/lib/')
    shutil.copy(src[0], target)
    return findThunderEgg()


def findThunderJar():
    thunderdir = os.path.dirname(os.path.realpath(thunder.__file__))
    thunderJar = os.path.join(thunderdir, 'lib', 'thunder_2.10-'+str(thunder.__version__)+'.jar')
    if not os.path.isfile(thunderJar):
        raise Exception("Thunder jar file not found at '%s'. Does Thunder need to be rebuilt?")
    return thunderJar


def getCommaSeparatedOptionsList(childOptionsFlag, commaSeparated, additionalDefault=None):
    lst = commaSeparated.split(",") if commaSeparated else []
    if additionalDefault:
        lst.append(additionalDefault)
    if lst:
        return [childOptionsFlag, ",".join(lst)]
    return []


def transformArguments(args):
    """
    Modifies command line arguments passed to the thunder launch scripts
    by adding required jar and egg files (if not already present).

    The passed arguments will be modified as follows and returned in a new list:
    1. the first element of args will be dropped (assumption is that this is sys.argv[0],
    the program name, which the caller is responsible for replacing).
    2. if not already present, the flag "--master" will be added to the arg list, with a
    value taken from the MASTER environment variable if present, otherwise generated
    from the DNS name of the master node if we believe we are being run on ec2, and
    otherwise set to "local".
    3. the flags "--py-files" and "--jars" will have the thunder egg file and thunder
    jar file appended to the comma-separated list value if present in the arg list.
    If they are not already in the arg list, they will be added with the egg and jar
    as their values.

    As a side effect, if the "master" arg is set to anything other than "local*",
    the thunder egg file will be rebuilt via a system call to setup.py if it does
    not already exist.

    Parameters
    ----------
    args: sequence of strings
        Sequence of string arguments, as would obtained from sys.argv

    Returns
    -------
        Sequence of string arguments
    """
    # check for a few specific flags and cache their values; put other
    # arguments in a passthrough list
    opts = {}
    passthruArgs = []
    if len(args) > 1:
        done = object()  # sentinel
        argIter = iter(args[1:])
        nextArg = next(argIter, done)
        while nextArg is not done:
            strippedArg = nextArg.lower().lstrip("-")
            if strippedArg in frozenset(["master", "py-files", "jars", "driver-class-path"]):
                opts[strippedArg] = next(argIter, "")
            else:
                passthruArgs.append(nextArg)
            nextArg = next(argIter, done)

    # look for and rebuild the thunder egg file if necessary
    master = getMasterURI(opts)
    if "local" not in master:
        thunderEgg = findThunderEgg()
        if not thunderEgg:
            thunderEgg = buildThunderEgg()
            if not thunderEgg:
                raise Exception("Unable to find or rebuild thunder .egg file.")
    else:
        thunderEgg = None

    thunderJar = '"%s"' % findThunderJar()

    # update arguments list with new values
    pyFiles = getCommaSeparatedOptionsList("--py-files", opts.get("py-files", []), thunderEgg)
    jars = getCommaSeparatedOptionsList("--jars", opts.get("jars", []), thunderJar)
    driverJars = getCommaSeparatedOptionsList("--driver-class-path", opts.get("driver-class-path", []), thunderJar)

    retVals = ["--master", master]
    retVals.extend(pyFiles)
    retVals.extend(jars)
    retVals.extend(driverJars)
    retVals.extend(passthruArgs)

    return retVals


def getFilteredHelpMessage(wrappedScriptPath, usage):
    msg = usage
    p = subprocess.Popen([wrappedScriptPath, "-h"], stderr=subprocess.PIPE)
    _, errOut = p.communicate()
    msg += '\n'.join([line for line in errOut.split('\n') if not line.lower().startswith("usage")])
    return msg