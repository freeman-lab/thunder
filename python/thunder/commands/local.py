#!/usr/bin/env python
import os
import sys

try:
    import thunder
except ImportError as e:
    thunder = None
    print("Unable to import Thunder, this is likely due to an incorrect PYTHONPATH or missing dependencies.\nGot the following error during import:\n%s" % e)
    sys.exit()
except SyntaxError as e:
    thunder = None
    print("Unable to import Thunder, got the following syntax error:\n%s" % e)
    sys.exit()
except:
    print("Unable to import Thunder for unknown reasons, got the following error:\n%s" % e)
    sys.exit()

from thunder.utils.launch import getFilteredHelpMessage, getSparkHome, transformArguments


def getUsage(wrappedScriptName='pyspark'):
    scriptName = os.path.basename(sys.argv[0])
    return "Usage: %s [%s options]\n" % (sys.argv[0], wrappedScriptName) + \
           "The '%s' script is a wrapper around Spark's '%s', and accepts all the same options, " % (scriptName, wrappedScriptName) + \
           "although not all are meaningful when running python scripts. Options for '%s' follow.\n" % wrappedScriptName


def main():
    SPARK_HOME = getSparkHome()

    childArgs = transformArguments(sys.argv)

    sparkSubmit = os.path.join(SPARK_HOME, 'bin', 'pyspark')

    # add python script
    os.environ['PYTHONSTARTUP'] = os.path.join(os.path.dirname(os.path.realpath(thunder.__file__)), 'utils', 'shell.py')

    # check for ipython option
    if "-i" in childArgs:
        os.environ['IPYTHON'] = '1'

    # check for notebook option
    if "-n" in childArgs:
        os.environ['IPYTHON_OPTS'] = 'notebook'

    # check for help flags, and print our own first if present
    if "-h" in childArgs or "--help" in childArgs:
        print >> sys.stderr, getFilteredHelpMessage(sparkSubmit, getUsage())
    else:
        os.execv(sparkSubmit, childArgs)

if __name__ == "__main__":
    main()
