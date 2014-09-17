import argparse
import os
import glob

from pyspark import SparkContext

from thunderdatatest import ThunderDataTest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run mllib performance tests")
    parser.add_argument("numtrials", type=int)
    parser.add_argument("datatype", type=str, choices=("create", "datafile"))
    parser.add_argument("persistencetype", type=str, choices=("memory", "disk", "none"))
    parser.add_argument("testname", type=str)
    parser.add_argument("--numrecords", type=int, default=1000, required=False)
    parser.add_argument("--numdims", type=int, default=5, required=False)
    parser.add_argument("--numpartitions", type=int, default=2, required=False)
    parser.add_argument("--numiterations", type=int, required=False)
    parser.add_argument("--savefile", type=str, default=None, required=False)
    parser.add_argument("--datafile", type=str, default=None, required=False)

    args = parser.parse_args()

    if args.datatype == "datafile":
        if args.datafile is None:
            raise ValueError("must specify a datafile location if datatype is datafile, use '--datafile myfile' ")

    if "save" in args.testname:
        if args.savefile is None:
            raise ValueError("must specify a savefile location if test includes saving, use '--savefile myfile' ")

    sc = SparkContext(appName="ThunderTestRunner: " + args.testname)

    test = ThunderDataTest.initialize(args.testname, sc)

    if args.datatype == "datafile":
        test.loadinputdata(args.datafile, args.savefile)
    elif args.datatype == "create":
        test.createinputdata(args.numrecords, args.numdims, args.numpartitions)

    results = test.run(args.numtrials, args.persistencetype)

    print("results: " + str(results))
    print("minimum: " + str(min(results)))

