""" Simple wrapper for a Spark Context to provide loading functionality """

import os

from thunder.utils.common import checkParams, handleFormat, raiseErrorIfPathExists
from thunder.utils.datasets import DataSets
from thunder.utils.params import Params


class ThunderContext():
    """
    Wrapper for a SparkContext that provides functionality for loading data.

    Also supports creation of example datasets, and loading example
    data both locally and from EC2.

    Attributes
    ----------
    `_sc` : SparkContext
        Spark context for Spark functionality

    `_credentials` : AWSCredentials object, optional, default = None
        Stores public and private keys for AWS services. Typically available through
        configuration files, and but can optionally be set on the ThunderContext.
        See setAWSCredentials().
    """

    def __init__(self, sparkcontext):
        self._sc = sparkcontext
        self._credentials = None

    @classmethod
    def start(cls, *args, **kwargs):
        """Starts a ThunderContext using the same arguments as SparkContext"""
        from pyspark import SparkContext
        return ThunderContext(SparkContext(*args, **kwargs))

    def loadSeries(self, dataPath, nkeys=None, nvalues=None, inputFormat='binary', minPartitions=None,
                   confFilename='conf.json', keyType=None, valueType=None):
        """
        Loads a Series object from data stored as text or binary files.

        Supports single files or multiple files stored on a local file system, a networked file system (mounted
        and available on all cluster nodes), Amazon S3, or HDFS.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename. Examples
            of valid dataPaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        nkeys: int, optional (but required if `inputFormat` is 'text')
            dimensionality of data keys. (For instance, (x,y,z) keyed data for 3-dimensional image timeseries data.)
            For text data, number of keys must be specified in this parameter; for binary data, number of keys must be
            specified either in this parameter or in a configuration file named by the 'conffile' argument if this
            parameter is not set.

        nvalues: int, optional (but required if `inputFormat` is 'text')
            Number of values expected to be read. For binary data, nvalues must be specified either in this parameter
            or in a configuration file named by the 'conffile' argument if this parameter is not set.

        inputFormat: {'text', 'binary'}. optional, default 'binary'
            Format of data to be read.

        minPartitions: int, optional
            Explicitly specify minimum number of Spark partitions to be generated from this data. Used only for
            text data. Default is to use minParallelism attribute of Spark context object.

        confFilename: string, optional, default 'conf.json'
            Path to JSON file with configuration options including 'nkeys', 'nvalues', 'keytype', and 'valuetype'.
            If a file is not found at the given path, then the base directory given in 'datafile'
            will also be checked. Parameters `nkeys` or `nvalues` that are specified as explicit arguments to this
            method will take priority over those found in conffile if both are present.

        Returns
        -------
        data: thunder.rdds.Series
            A newly-created Series object, wrapping an RDD of series data. This RDD will have as keys an n-tuple
            of int, with n given by `nkeys` or the configuration passed in `conffile`. RDD values will be a numpy
            array of length `nvalues` (or as specified in the passed configuration file).
        """
        checkParams(inputFormat, ['text', 'binary'])

        from thunder.rdds.fileio.seriesloader import SeriesLoader
        loader = SeriesLoader(self._sc, minPartitions=minPartitions)

        if inputFormat.lower() == 'text':
            data = loader.fromText(dataPath, nkeys=nkeys)
        else:
            # must be either 'text' or 'binary'
            data = loader.fromBinary(dataPath, confFilename=confFilename, nkeys=nkeys, nvalues=nvalues,
                                     keyType=keyType, valueType=valueType)
        return data

    def loadImages(self, dataPath, dims=None, inputFormat='stack', ext=None, dtype=None,
                   startIdx=None, stopIdx=None, recursive=False, nplanes=None, npartitions=None,
                   renumber=False, confFilename='conf.json'):
        """
        Loads an Images object from data stored as a binary image stack, tif, or png files.

        Supports single files or multiple files, stored on a local file system, a networked file sytem
        (mounted and available on all nodes), or Amazon S3. HDFS is not currently supported for image file data.

        Parameters that are not explicitly passed as keyword arguments may be read out from an optional configuration
        JSON file located in the same directory as the data files.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename. Examples
            of valid dataPaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        dims: tuple of positive int, optional
            Dimensions of input image data, for instance (1024, 1024, 48). Binary stack data will be interpreted as
            coming from a multidimensional array of the specified dimensions.

            Dimensions are required for 'stack' inputFormat, either as an explicit keyword argument or in a
            configuration JSON file. If inputFormat is 'png' or 'tif', the dims parameter (if any) will be ignored;
            data dimensions will instead be read out from the image file headers.

            Stack data should be stored in row-major order (Fortran / Matlab convention) rather than column-major order
            (C or python/numpy convention), where the first dimension corresponds to that which is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, zo), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)].

        inputFormat: {'stack', 'png', 'tif'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data. 'png' or 'tif' indicate
            image files of the corresponding formats. Each page of a multipage tif file will be interpreted as a
            separate z-plane. For all formats, separate files are interpreted as distinct time points, with ordering
            given by lexicographic sorting of file names.

        ext: string, optional, default None
            Extension required on data files to be loaded. By default will be "stack" if inputFormat=="stack", "tif" for
            inputFormat=='tif', and 'png' for inputFormat="png".

        dtype: string dtype specifier or numpy dtype object, optional.
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputFormat is
            'tif' or 'png', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            image file headers. If inputFormat is 'stack', then dtype will be assumed to be 'int16' if it is not specified
            either as a keyword parameter or in a configuration json file.

        startIdx: nonnegative int, optional
            startIdx and stopIdx are convenience parameters to allow only a subset of input files to be read in. These
            parameters give the starting index (inclusive) and final index (exclusive) of the data files to be used
            after lexicographically sorting all input data files matching the dataPath argument. For example,
            startIdx=None (the default) and stopIdx=10 will cause only the first 10 data files in dataPath to be read
            in; startIdx=2 and stopIdx=3 will cause only the third file (zero-based index of 2) to be read in. startIdx
            and stopIdx use the python slice indexing convention (zero-based indexing with an exclusive final position).

        stopIdx: nonnegative int, optional
            See startIdx.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at dataPath, loading all files in the tree that
            have an appropriate extension. Recursive loading is currently only implemented for local filesystems
            (not s3).

        nplanes: positive integer, default None
            If passed, will cause a single image file to be subdivided into multiple records. Every
            `nplanes` z-planes (or multipage tif pages) in the file will be taken as a new record, with the
            first nplane planes of the first file being record 0, the second nplane planes being record 1, etc,
            until the first file is exhausted and record ordering continues with the first nplane planes of the
            second file, and so on. With nplanes=None (the default), a single file will be considered as
            representing a single record. Keys are calculated assuming that all input files contain the same
            number of records; if the number of records per file is not the same across all files,
            then `renumber` should be set to True to ensure consistent keys.

        npartitions: positive int, optional
            If specified, request a certain number of partitions for the underlying Spark RDD. Default is 1
            partition per image file.

        renumber: boolean, optional, default False
            If renumber evaluates to True, then the keys for each record will be explicitly recalculated after
            all images are loaded. This should only be necessary at load time when different files contain
            different number of records. See Images.renumber().

        confFilename: string, optional, default 'conf.json'
            Path to optional JSON file with configuration options such as 'dims', 'dtype', 'ext', 'recursive', and
            'nplanes'. confFilename may be an absolute or a relative path; relative paths will be interpreted
            relative to the base directory given in 'dataPath'. If a configuration file is found, then parameters given
            in this file will be used unless they are overridden by explicit keyword arguments. Not all parameters
            are used by all inputFormats; see documentation for individual parameters above.

        Returns
        -------
        data: thunder.rdds.Images
            A newly-created Images object, wrapping an RDD of <int index, numpy array> key-value pairs.

        """
        checkParams(inputFormat, ['stack', 'png', 'tif', 'tif-stack'])

        from thunder.rdds.fileio.imagesloader import ImagesLoader
        loader = ImagesLoader(self._sc)

        if inputFormat.lower() == 'stack':
            data = loader.fromStack(dataPath, dims=dims, dtype=dtype, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                    recursive=recursive, nplanes=nplanes, npartitions=npartitions,
                                    confFilename=confFilename)
        elif inputFormat.lower().startswith('tif'):
            data = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx, recursive=recursive,
                                  nplanes=nplanes, npartitions=npartitions, confFilename=confFilename)
        else:
            if nplanes:
                raise NotImplementedError("nplanes argument is not supported for png files")
            data = loader.fromPng(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                  recursive=recursive, npartitions=npartitions, confFilename=confFilename)

        if not renumber:
            return data
        else:
            return data.renumber()

    def loadImagesAsSeries(self, dataPath, dims=None, inputFormat='stack', ext=None, dtype='int16',
                           blockSize="150M", blockSizeUnits="pixels", startIdx=None, stopIdx=None,
                           shuffle=True, recursive=False, nplanes=None, npartitions=None,
                           renumber=False, confFilename='conf.json'):
        """
        Load Images data as Series data.

        Image parameters that are not explicitly passed as keyword arguments may be read out from an optional
        configuration JSON file located in the same directory as the data files (when shuffle=True). Parameters
        that may be passed in a configuration file include 'dims', 'ext', 'dtype', 'recursive', and 'nplanes'.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename. Examples
            of valid dataPaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        dims: tuple of positive int, optional (but required if inputFormat is 'stack')
            Dimensions of input image data, for instance (1024, 1024, 48). Binary stack data will be interpreted as
            coming from a multidimensional array of the specified dimensions.

            Dimensions are required for 'stack' inputFormat, either as an explicit keyword argument or in a
            configuration JSON file. If inputFormat is 'tif', the dims parameter (if any) will be ignored;
            data dimensions will instead be read out from the image file headers.

            Stack data should be stored in row-major order (Fortran / Matlab convention) rather than column-major order
            (C or python/numpy convention), where the first dimension corresponds to that which is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, zo), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)].

        inputFormat: {'stack', 'tif'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data, while 'tif' indicates
            greyscale / luminance TIF images. Each page of a multipage tif file will be interpreted as a separate
            z-plane. For both stacks and tif stacks, separate files are interpreted as distinct time points, with
            ordering given by lexicographic sorting of file names.

        ext: string, optional, default None
            Extension required on data files to be loaded. By default will be "stack" if inputFormat=="stack", "tif" for
            inputFormat=='tif'.

        dtype: string dtype specifier or numpy dtype object, optional.
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputFormat is
            'tif', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            image file headers. If inputFormat is 'stack', then dtype will be assumed to be 'int16' if it is not specified
            either as a keyword parameter or in a configuration json file.

        blockSize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of individual output files in bytes (or kilobytes, megabytes, gigabytes). If shuffle=True,
            blockSize can also be a tuple of int specifying either the number of pixels or of splits per dimension to
            apply to the loaded images, or an instance of BlockingStrategy. Whether a tuple of int is interpreted as
            pixels or as splits depends on the value of the blockSizeUnits parameter. blockSize also indirectly
            controls the number of Spark partitions to be used, with one partition used per block created.

        blockSizeUnits: string, either "pixels" or "splits" (or unique prefix of each, such as "s"), default "pixels"
            Specifies units to be used in interpreting a tuple passed as blockSizeSpec when shuffle=True. If a string
            or a BlockingStrategy instance is passed as blockSizeSpec, or if shuffle=False, this parameter has no
            effect.

        startIdx: nonnegative int, optional
            startIdx and stopIdx are convenience parameters to allow only a subset of input files to be read in. These
            parameters give the starting index (inclusive) and final index (exclusive) of the data files to be used
            after lexicographically sorting all input data files matching the dataPath argument. For example,
            startIdx=None (the default) and stopIdx=10 will cause only the first 10 data files in dataPath to be read
            in; startIdx=2 and stopIdx=3 will cause only the third file (zero-based index of 2) to be read in. startIdx
            and stopIdx use the python slice indexing convention (zero-based indexing with an exclusive final position).

        stopIdx: nonnegative int, optional
            See startIdx.

        shuffle: boolean, optional, default True
            Controls whether the conversion from Images to Series formats will make use of a Spark shuffle-based method.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at dataPath, loading all files in the tree that
            have an appropriate extension. Recursive loading is currently only implemented for local filesystems
            (not s3), and only with shuffle=True.

        nplanes: positive integer, default None
            If passed, will cause a single image file to be subdivided into multiple records. Every
            `nplanes` z-planes (or multipage tif pages) in the file will be taken as a new record, with the
            first nplane planes of the first file being record 0, the second nplane planes being record 1, etc,
            until the first file is exhausted and record ordering continues with the first nplane planes of the
            second file, and so on. With nplanes=None (the default), a single file will be considered as
            representing a single record. Keys are calculated assuming that all input files contain the same
            number of records; if the number of records per file is not the same across all files,
            then `renumber` should be set to True to ensure consistent keys. nplanes is only supported for
            shuffle=True (the default).

        npartitions: positive int, optional
            If specified, request a certain number of partitions for the underlying Spark RDD. Default is 1
            partition per image file. Only applies when shuffle=True.

        renumber: boolean, optional, default False
            If renumber evaluates to True, then the keys for each record will be explicitly recalculated after
            all images are loaded. This should only be necessary at load time when different files contain
            different number of records. renumber is only supported for shuffle=True (the default). See
            Images.renumber().

        confFilename: string, optional, default 'conf.json'
            Path to optional JSON file with configuration options such as 'dims', 'dtype', 'ext', 'recursive', and
            'nplanes'. confFilename may be an absolute or a relative path; relative paths will be interpreted
            relative to the base directory given in 'dataPath'. If a configuration file is found, then parameters given
            in this file will be used unless they are overridden by explicit keyword arguments. Not all parameters
            are used by all inputFormats; see documentation for individual parameters above. JSON configuration files
            are only supported for shuffle=True (the default).

        Returns
        -------
        data: thunder.rdds.Series
            A newly-created Series object, wrapping an RDD of timeseries data generated from the images in dataPath.
            This RDD will have as keys an n-tuple of int, with n given by the dimensionality of the original images. The
            keys will be the zero-based spatial index of the timeseries data in the RDD value. The value will be
            a numpy array of length equal to the number of image files loaded. Each loaded image file will contribute
            one point to this value array, with ordering as implied by the lexicographic ordering of image file names.
        """
        checkParams(inputFormat, ['stack', 'tif', 'tif-stack'])

        if shuffle:
            from thunder.rdds.fileio.imagesloader import ImagesLoader
            loader = ImagesLoader(self._sc)
            if inputFormat.lower() == 'stack':
                images = loader.fromStack(dataPath, dims, dtype=dtype, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                          recursive=recursive, nplanes=nplanes, npartitions=npartitions,
                                          confFilename=confFilename)
            else:
                # tif / tif stack
                images = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                        recursive=recursive, nplanes=nplanes, npartitions=npartitions,
                                        confFilename=confFilename)
            if renumber:
                images = images.renumber()
            return images.toBlocks(blockSize, units=blockSizeUnits).toSeries()

        else:
            from thunder.rdds.fileio.seriesloader import SeriesLoader
            if nplanes is not None:
                raise NotImplementedError("nplanes is not supported with shuffle=False")
            if npartitions is not None:
                raise NotImplementedError("npartitions is not supported with shuffle=False")
            if renumber:
                raise NotImplementedError("renumber is not supported with shuffle=False")
            if confFilename != "conf.json":
                raise NotImplementedError("JSON configuration files are not supported with shuffle=False")

            if not ext:
                ext = DEFAULT_EXTENSIONS.get(inputFormat.lower(), None)

            loader = SeriesLoader(self._sc)
            if inputFormat.lower() == 'stack':
                return loader.fromStack(dataPath, dims, ext=ext, dtype=dtype, blockSize=blockSize,
                                        startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)
            else:
                # tif / tif stack
                return loader.fromTif(dataPath, ext=ext, blockSize=blockSize,
                                      startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)

    def convertImagesToSeries(self, dataPath, outputDirPath, dims=None, inputFormat='stack', ext=None,
                              dtype=None, blockSize="150M", blockSizeUnits="pixels", startIdx=None, stopIdx=None,
                              shuffle=True, overwrite=False, recursive=False, nplanes=None, npartitions=None,
                              renumber=False, confFilename='conf.json'):
        """
        Write out Images data as Series data, saved in a flat binary format.

        The resulting Series data files may subsequently be read in using the loadSeries() method. The Series data
        object that results will be equivalent to that which would be generated by loadImagesAsSeries(). It is expected
        that loading Series data directly from the series flat binary format, using loadSeries(), will be faster than
        converting image data to a Series object through loadImagesAsSeries().

        Image parameters that are not explicitly passed as keyword arguments may be read out from an optional
        configuration JSON file located in the same directory as the data files (when shuffle=True). Parameters
        that may be passed in a configuration file include 'dims', 'ext', 'dtype', 'recursive', and 'nplanes'.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename. Examples
            of valid dataPaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        outputDirPath: string
            Path to a directory into which to write Series file output. An outputdir argument may be either a path
            on the local file system or a URI-like format, as in dataPath. Examples of valid outputDirPaths include
            "a/relative/directory/", "s3n:///my-s3-bucket/data/myoutput/", or "file:///mnt/a/new/directory/". If the
            directory specified by outputDirPath already exists and the 'overwrite' parameter is False, this method
            will throw a ValueError. If the directory exists and 'overwrite' is True, the existing directory and all
            its contents will be deleted and overwritten.

        dims: tuple of positive int, optional (but required if inputFormat is 'stack')
            Dimensions of input image data, for instance (1024, 1024, 48). Binary stack data will be interpreted as
            coming from a multidimensional array of the specified dimensions.

            Dimensions are required for 'stack' inputFormat, either as an explicit keyword argument or in a
            configuration JSON file. If inputFormat is 'tif', the dims parameter (if any) will be ignored;
            data dimensions will instead be read out from the image file headers.

            Stack data should be stored in row-major order (Fortran / Matlab convention) rather than column-major order
            (C or python/numpy convention), where the first dimension corresponds to that which is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, zo), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)].

        inputFormat: {'stack', 'tif'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data, while 'tif' indicates
            greyscale / luminance TIF images. Each page of a multipage tif file will be interpreted as a separate
            z-plane. For both stacks and tif stacks, separate files are interpreted as distinct time points, with
            ordering given by lexicographic sorting of file names.

        ext: string, optional, default None
            Extension required on data files to be loaded. By default will be "stack" if inputFormat=="stack", "tif" for
            inputFormat=='tif'.

        dtype: string dtype specifier or numpy dtype object, optional.
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputFormat is
            'tif', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            image file headers. If inputFormat is 'stack', then dtype will be assumed to be 'int16' if it is not specified
            either as a keyword parameter or in a configuration json file.

        blockSize: string formatted as e.g. "64M", "512k", "2G", or positive int, tuple of positive int, or instance of
                   BlockingStrategy. optional, default "150M"
            Requested size of individual output files in bytes (or kilobytes, megabytes, gigabytes). blockSize can also
            be an instance of blockingStrategy, or a tuple of int specifying either the number of pixels or of splits
            per dimension to apply to the loaded images. Whether a tuple of int is interpreted as pixels or as splits
            depends on the value of the blockSizeUnits parameter.  This parameter also indirectly controls the number
            of Spark partitions to be used, with one partition used per block created.

        blockSizeUnits: string, either "pixels" or "splits" (or unique prefix of each, such as "s"), default "pixels"
            Specifies units to be used in interpreting a tuple passed as blockSizeSpec when shuffle=True. If a string
            or a BlockingStrategy instance is passed as blockSizeSpec, or if shuffle=False, this parameter has no
            effect.

        startIdx: nonnegative int, optional
            startIdx and stopIdx are convenience parameters to allow only a subset of input files to be read in. These
            parameters give the starting index (inclusive) and final index (exclusive) of the data files to be used
            after lexicographically sorting all input data files matching the dataPath argument. For example,
            startIdx=None (the default) and stopIdx=10 will cause only the first 10 data files in dataPath to be read
            in; startIdx=2 and stopIdx=3 will cause only the third file (zero-based index of 2) to be read in. startIdx
            and stopIdx use the python slice indexing convention (zero-based indexing with an exclusive final position).

        stopIdx: nonnegative int, optional
            See startIdx.

        shuffle: boolean, optional, default True
            Controls whether the conversion from Images to Series formats will make use of a Spark shuffle-based method.

        overwrite: boolean, optional, default False
            If true, the directory specified by outputDirPath will first be deleted, along with all its contents, if it
            already exists. (Use with caution.) If false, a ValueError will be thrown if outputDirPath is found to
            already exist.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at dataPath, loading all files in the tree that
            have an appropriate extension. Recursive loading is currently only implemented for local filesystems
            (not s3), and only with shuffle=True.

        nplanes: positive integer, default None
            If passed, will cause a single image file to be subdivided into multiple records. Every
            `nplanes` z-planes (or multipage tif pages) in the file will be taken as a new record, with the
            first nplane planes of the first file being record 0, the second nplane planes being record 1, etc,
            until the first file is exhausted and record ordering continues with the first nplane planes of the
            second file, and so on. With nplanes=None (the default), a single file will be considered as
            representing a single record. Keys are calculated assuming that all input files contain the same
            number of records; if the number of records per file is not the same across all files,
            then `renumber` should be set to True to ensure consistent keys. nplanes is only supported for
            shuffle=True (the default).

        npartitions: positive int, optional
            If specified, request a certain number of partitions for the underlying Spark RDD. Default is 1
            partition per image file. Only applies when shuffle=True.

        renumber: boolean, optional, default False
            If renumber evaluates to True, then the keys for each record will be explicitly recalculated after
            all images are loaded. This should only be necessary at load time when different files contain
            different number of records. renumber is only supported for shuffle=True (the default). See
            Images.renumber().

        confFilename: string, optional, default 'conf.json'
            Path to optional JSON file with configuration options such as 'dims', 'dtype', 'ext', 'recursive', and
            'nplanes'. confFilename may be an absolute or a relative path; relative paths will be interpreted
            relative to the base directory given in 'dataPath'. If a configuration file is found, then parameters given
            in this file will be used unless they are overridden by explicit keyword arguments. Not all parameters
            are used by all inputFormats; see documentation for individual parameters above. JSON configuration files
            are only supported for shuffle=True (the default).
        """
        checkParams(inputFormat, ['stack', 'tif', 'tif-stack'])

        if not overwrite:
            raiseErrorIfPathExists(outputDirPath, awsCredentialsOverride=self._credentials)
            overwrite = True  # prevent additional downstream checks for this path

        if shuffle:
            from thunder.rdds.fileio.imagesloader import ImagesLoader
            loader = ImagesLoader(self._sc)
            if inputFormat.lower() == 'stack':
                images = loader.fromStack(dataPath, dims, ext=ext, dtype=dtype, startIdx=startIdx, stopIdx=stopIdx,
                                          recursive=recursive, nplanes=nplanes, npartitions=npartitions,
                                          confFilename=confFilename)
            else:
                # 'tif' or 'tif-stack'
                images = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                        recursive=recursive, nplanes=nplanes, npartitions=npartitions,
                                        confFilename=confFilename)
            if renumber:
                images = images.renumber()
            images.toBlocks(blockSize, units=blockSizeUnits).saveAsBinarySeries(outputDirPath, overwrite=overwrite)
        else:
            from thunder.rdds.fileio.seriesloader import SeriesLoader
            if nplanes is not None:
                raise NotImplementedError("nplanes is not supported with shuffle=False")
            if npartitions is not None:
                raise NotImplementedError("npartitions is not supported with shuffle=False")
            if renumber:
                raise NotImplementedError("renumber is not supported with shuffle=False")
            if confFilename != "conf.json":
                raise NotImplementedError("JSON configuration files are not supported with shuffle=False")
            # look up default extension for series loader; image loader will look in conf file
            if not ext:
                ext = DEFAULT_EXTENSIONS.get(inputFormat.lower(), None)

            loader = SeriesLoader(self._sc)
            if inputFormat.lower() == 'stack':
                loader.saveFromStack(dataPath, outputDirPath, dims, ext=ext, dtype=dtype,
                                     blockSize=blockSize, overwrite=overwrite, startIdx=startIdx,
                                     stopIdx=stopIdx, recursive=recursive)
            else:
                # 'tif' or 'tif-stack'
                loader.saveFromTif(dataPath, outputDirPath, ext=ext, blockSize=blockSize,
                                   startIdx=startIdx, stopIdx=stopIdx, overwrite=overwrite,
                                   recursive=recursive)

    def makeExample(self, dataset, **opts):
        """
        Make an example data set for testing analyses.

        Options include 'pca', 'kmeans', and 'ica'.
        See thunder.utils.datasets for detailed options.

        Parameters
        ----------
        dataset : str
            Which dataset to generate

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset

        """
        checkParams(dataset, ['kmeans', 'pca', 'ica'])

        return DataSets.make(self._sc, dataset, **opts)

    def loadExample(self, dataset=None):
        """
        Load a local example data set for testing analyses.

        Some of these data sets are extremely downsampled and should be considered
        useful only for testing the API. If called with None,
        will return list of available datasets.

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : Data object
            Generated dataset as a Thunder data objects (e.g Series or Images)
        """
        import atexit
        import shutil
        import tempfile
        from pkg_resources import resource_listdir, resource_filename

        DATASETS = {
            'iris': 'iris',
            'fish-series': 'fish/bin',
            'fish-images': 'fish/tif-stack'
        }

        if dataset is None:
            return DATASETS.keys()

        checkParams(dataset, DATASETS.keys())

        if 'ec2' in self._sc.master:
            tmpdir = os.path.join('/root/thunder/python/thunder/utils', 'data', DATASETS[dataset])
        else:
            tmpdir = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, tmpdir)

            def copyLocal(target):
                files = resource_listdir('thunder.utils.data', target)
                for f in files:
                    path = resource_filename('thunder.utils.data', os.path.join(target, f))
                    shutil.copy(path, tmpdir)

            copyLocal(DATASETS[dataset])

        if dataset == "iris":
            return self.loadSeries(tmpdir)
        elif dataset == "fish-series":
            return self.loadSeries(tmpdir).astype('float')
        elif dataset == "fish-images":
            return self.loadImages(tmpdir, inputFormat="tif")

    def loadExampleS3(self, dataset=None):
        """
        Load an example data set from S3.

        Info on the included datasets can be found at the CodeNeuro data repository
        (http://datasets.codeneuro.org/). If called with None, will return
        list of available datasets.

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : a Data object (usually a Series or Images)
            The dataset as one of Thunder's data objects

        params : dict
            Parameters or metadata for dataset
        """
        DATASETS = {
            'ahrens.lab/direction.selectivity': 'ahrens.lab/direction.selectivity/1/',
            'ahrens.lab/optomotor.response': 'ahrens.lab/optomotor.response/1/',
            'svoboda.lab/tactile.navigation': 'svoboda.lab/tactile.navigation/1/'
        }

        if 'local' in self._sc.master:
            raise Exception("Must be running on an EC2 cluster to load this example data set")

        if dataset is None:
            return DATASETS.keys()

        checkParams(dataset, DATASETS.keys())

        basePath = 's3n://neuro.datasets/'
        dataPath = DATASETS[dataset]

        data = self.loadSeries(basePath + dataPath + 'series')
        params = self.loadParams(basePath + dataPath + 'params/covariates.json')

        return data, params

    def loadParams(self, path):
        """
        Load a file with parameters from a local file system or S3.

        Assumes file is JSON with basic types (strings, integers, doubles, lists),
        in either a single dict or list of dict-likes, and each dict has at least
        a "name" field and a "value" field.

        Useful for loading generic meta data, parameters, covariates, etc.

        Parameters
        ----------
        path : str
            Path to file, can be on a local file system or an S3 bucket

        Returns
        -------
        A dict or list with the parameters
        """
        import json
        from thunder.rdds.fileio.readers import getFileReaderForPath, FileNotFoundError

        reader = getFileReaderForPath(path)(awsCredentialsOverride=self._credentials)
        try:
            buffer = reader.read(path)
        except FileNotFoundError:
            raise Exception("Cannot find file %s" % path)

        return Params(json.loads(buffer))

    def loadSeriesLocal(self, dataFilePath, inputFormat='npy', minPartitions=None, keyFilePath=None, varName=None):
        """
        Load a Series object from a local file (either npy or MAT format).

        File should contain a 1d or 2d matrix, where each row
        of the input matrix is a record.

        Keys can be provided in a separate file (with variable name 'keys', for MAT files).
        If not provided, linear indices will be used for keys.

        Parameters
        ----------
        dataFilePath: str
            File to import

        varName : str, optional, default = None
            Variable name to load (for MAT files only)

        keyFilePath : str, optional, default = None
            File containing the keys for each record as another 1d or 2d array

        minPartitions : Int, optional, default = 1
            Number of partitions for RDD
        """
        checkParams(inputFormat, ['mat', 'npy'])

        from thunder.rdds.fileio.seriesloader import SeriesLoader
        loader = SeriesLoader(self._sc, minPartitions=minPartitions)

        if inputFormat.lower() == 'mat':
            if varName is None:
                raise Exception('Must provide variable name for loading MAT files')
            data = loader.fromMatLocal(dataFilePath, varName, keyFilePath)
        else:
            data = loader.fromNpyLocal(dataFilePath, keyFilePath)

        return data

    def export(self, data, filename, format=None, overwrite=False, varname=None):
        """
        Export local array data to a variety of formats.

        Can write to a local file sytem or S3 (destination inferred from filename schema).
        S3 writing useful for persisting arrays when working in an environment without
        accessible local storage.

        Parameters
        ----------
        data : array-like
            The data to export

        filename : str
            Output location (path/to/file.ext)

        format : str, optional, default = None
            Ouput format ("npy", "mat", or "txt"), if not provided will
            try to infer from file extension.

        overwrite : boolean, optional, default = False
            Whether to overwrite if directory or file already exists

        varname : str, optional, default = None
            Variable name for writing "mat" formatted files
        """
        from numpy import save, savetxt
        from scipy.io import savemat
        from StringIO import StringIO

        from thunder.rdds.fileio.writers import getFileWriterForPath

        path, file, format = handleFormat(filename, format)
        checkParams(format, ["npy", "mat", "txt"])
        clazz = getFileWriterForPath(filename)
        writer = clazz(path, file, overwrite=overwrite, awsCredentialsOverride=self._credentials)

        stream = StringIO()

        if format == "mat":
            varname = os.path.splitext(file)[0] if varname is None else varname
            savemat(stream, mdict={varname: data}, oned_as='column', do_compression='true')
        if format == "npy":
            save(stream, data)
        if format == "txt":
            savetxt(stream, data)

        stream.seek(0)
        writer.writeFile(stream.buf)

    def setAWSCredentials(self, awsAccessKeyId, awsSecretAccessKey):
        """
        Manually set AWS access credentials to be used by Thunder.

        This method is provided primarily for hosted environments that do not provide
        filesystem access (e.g. Databricks Cloud). Typically AWS credentials can be set
        and read from core-site.xml (for Hadoop input format readers, such as Series
        binary files), ~/.boto or other boto credential file locations, or the environment
        variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY. These credentials should
        be configured automatically in clusters launched by the thunder-ec2 script, and
        so this method should not have to be called.

        Parameters
        ----------
        awsAccessKeyId : string
             AWS public key, usually starts with "AKIA"

        awsSecretAccessKey : string
            AWS private key
        """
        from thunder.utils.common import AWSCredentials
        self._credentials = AWSCredentials(awsAccessKeyId, awsSecretAccessKey)
        self._credentials.setOnContext(self._sc)


DEFAULT_EXTENSIONS = {
    "stack": "stack",
    "tif": "tif",
    "tif-stack": "tif",
    "png": "png",
    "mat": "mat",
    "npy": "npy",
    "txt": "txt"
}