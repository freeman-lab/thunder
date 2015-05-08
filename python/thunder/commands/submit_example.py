#!/usr/bin/env python
import glob
import os
import sys

try:
    import thunder
except ImportError as e:
    thunder = None
    print("Unable to import Thunder, this is likely due to an incorrect PYTHONPATH or missing dependencies.\nGot the following error during import:\n%s" % e)
    sys.exit()

from thunder.utils.launch import getFilteredHelpMessage, getSparkHome, transformArguments


def getStandaloneDir():
    return os.path.join(os.path.dirname(os.path.realpath(thunder.__file__)), "standalone")


def getExampleNames():
    standaloneDir = getStandaloneDir()
    return [os.path.basename(fname)[:-3] for fname in glob.glob(standaloneDir + os.sep + "*.py")]


def getUsage(wrappedScriptName='spark-submit'):
    scriptName = os.path.basename(sys.argv[0])
    return "Usage: %s [%s options] <analysis name> [analysis options]\n" % (sys.argv[0], wrappedScriptName) + \
           "'%s' runs analysis scripts that are found in the %s directory.\n" % (scriptName, getStandaloneDir()) + \
           "It is a wrapper around Spark's '%s', and accepts all the same options, " % wrappedScriptName + \
           "although not all are meaningful when running python scripts. Options for '%s' follow.\n" % wrappedScriptName


def main():
    SPARK_HOME = getSparkHome()
    sparkSubmit = os.path.join(SPARK_HOME, 'bin', 'spark-submit')

    if len(sys.argv) < 2:
        print >> sys.stderr, getFilteredHelpMessage(sparkSubmit, getUsage())
        exit(1)

    exampleNames = getExampleNames()

    found = False
    childArgs = transformArguments(sys.argv)
    for idx, arg in enumerate(childArgs):
        if arg in exampleNames:
            childArgs[idx] = os.path.join(getStandaloneDir(), arg+".py")
            found = True
            break
        elif arg == '-h' or arg == '--help':
            print >> sys.stderr, getFilteredHelpMessage(sparkSubmit, getUsage())
            exit(0)
    if not found:
        raise Exception("Did not recognize a known example analysis name; must be a script in %s (one of: %s)" %
                        (getStandaloneDir(), str(getExampleNames())[1:-1]))

    os.system(sparkSubmit + " " + " ".join(childArgs))

if __name__ == "__main__":
    main()
