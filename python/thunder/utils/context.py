""" Simple wrapper for a Spark Context to provide loading functionality """

from thunder.utils.datasets import DataSets
from thunder.utils.common import checkParams, raiseErrorIfPathExists


class ThunderContext():
    """
    Wrapper for a SparkContext that provides functionality for loading data.

    Also supports creation of example datasets, and loading example
    data both locally and from EC2.

    Attributes
    ----------
    `_sc` : SparkContext
        Spark context for Spark functionality

    awsAccessKeyId: None, or string
    awsSecretAccessKey: None, or string
        Public and private keys for AWS services. Typically the credentials should be accessible
        through any of several different configuration files, and so should not have to be set
        on the ThunderContext. See setAWSCredentials().
    """

    def __init__(self, sparkcontext):
        self._sc = sparkcontext
        self.awsCredentials = None

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
        loader = SeriesLoader(self._sc, minPartitions=minPartitions, awsCredentialsOverride=self.awsCredentials)

        if inputFormat.lower() == 'text':
            data = loader.fromText(dataPath, nkeys=nkeys)
        else:
            # must be either 'text' or 'binary'
            data = loader.fromBinary(dataPath, confFilename=confFilename, nkeys=nkeys, nvalues=nvalues,
                                     keyType=keyType, valueType=valueType)
        return data

    def loadImages(self, dataPath, dims=None, inputFormat='stack', ext=None, dtype='int16',
                   startIdx=None, stopIdx=None, recursive=False, nplanes=None, npartitions=None,
                   renumber=False):
        """
        Loads an Images object from data stored as a binary image stack, tif, or png files.

        Supports single files or multiple files, stored on a local file system, a networked file sytem
        (mounted and available on all nodes), or Amazon S3. HDFS is not currently supported for image file data.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename. Examples
            of valid dataPaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        dims: tuple of positive int, optional (but required if inputFormat is 'stack')
            Dimensions of input image data, similar to a numpy 'shape' parameter, for instance (1024, 1024, 48). Binary
            stack data will be interpreted as coming from a multidimensional array of the specified dimensions. Stack
            data should be stored in row-major order (Fortran or Matlab convention) rather than column-major order (C
            or python/numpy convention), where the first dimension corresponds to that which is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, zo), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)].
            If inputFormat is 'png' or 'tif', the dims parameter (if any) will be ignored; data dimensions
            will instead be read out from the image file headers.

        inputFormat: {'stack', 'png', 'tif'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data. 'png' or 'tif' indicate
            image files of the corresponding formats. Each page of a multipage tif file will be interpreted as a
            separate z-plane. For all formats, separate files are interpreted as distinct time points, with ordering
            given by lexicographic sorting of file names.

        ext: string, optional, default None
            Extension required on data files to be loaded. By default will be "stack" if inputFormat=="stack", "tif" for
            inputFormat=='tif', and 'png' for inputFormat="png".

        dtype: string or numpy dtype. optional, default 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputFormat is
            'tif' or 'png', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            tif headers.

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

        Returns
        -------
        data: thunder.rdds.Images
            A newly-created Images object, wrapping an RDD of <int index, numpy array> key-value pairs.

        """
        checkParams(inputFormat, ['stack', 'png', 'tif', 'tif-stack'])

        from thunder.rdds.fileio.imagesloader import ImagesLoader
        loader = ImagesLoader(self._sc, awsCredentialsOverride=self.awsCredentials)

        if not ext:
            ext = DEFAULT_EXTENSIONS.get(inputFormat.lower(), None)

        if inputFormat.lower() == 'stack':
            data = loader.fromStack(dataPath, dims, dtype=dtype, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                    recursive=recursive, nplanes=nplanes, npartitions=npartitions)
        elif inputFormat.lower().startswith('tif'):
            data = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx, recursive=recursive,
                                  nplanes=nplanes, npartitions=npartitions)
        else:
            if nplanes:
                raise NotImplementedError("nplanes argument is not supported for png files")
            data = loader.fromPng(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                  recursive=recursive, npartitions=npartitions)

        if not renumber:
            return data
        else:
            return data.renumber()

    def loadImagesAsSeries(self, dataPath, dims=None, inputFormat='stack', ext=None, dtype='int16',
                           blockSize="150M", blockSizeUnits="pixels", startIdx=None, stopIdx=None,
                           shuffle=True, recursive=False, nplanes=None, npartitions=None,
                           renumber=False):
        """
        Load Images data as Series data.

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

            The first dimension of the passed dims tuple should be the one that is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, z0), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)]. This is the opposite convention from that used by numpy,
            which by default has the fastest-changing dimension listed last (column-major convention). Thus, if loading
            a numpy array `ary`, where `ary.shape == (z, y, x)`, written to disk by `ary.tofile("myarray.stack")`, the
            corresponding dims parameter should be (x, y, z).
            If inputFormat is 'tif', the dims parameter (if any) will be ignored; data dimensions will instead
            be read out from the tif file headers.

        inputFormat: {'stack', 'tif'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data, while 'tif' indicates
            greyscale / luminance TIF images. Each page of a multipage tif file will be interpreted as a separate
            z-plane. For both stacks and tif stacks, separate files are interpreted as distinct time points, with
            ordering given by lexicographic sorting of file names.

        ext: string, optional, default None
            Extension required on data files to be loaded. By default will be "stack" if inputFormat=="stack", "tif" for
            inputFormat=='tif'.

        dtype: string or numpy dtype. optional, default 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputFormat is
            'tif', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            tif headers.

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

        if inputFormat.lower() == 'stack' and not dims:
            raise ValueError("Dimensions ('dims' parameter) must be specified if loading from binary image stack" +
                             " ('stack' value for 'inputFormat' parameter)")

        if not ext:
            ext = DEFAULT_EXTENSIONS.get(inputFormat.lower(), None)

        if shuffle:
            from thunder.rdds.fileio.imagesloader import ImagesLoader
            loader = ImagesLoader(self._sc, awsCredentialsOverride=self.awsCredentials)
            if inputFormat.lower() == 'stack':
                images = loader.fromStack(dataPath, dims, dtype=dtype, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                          recursive=recursive, nplanes=nplanes, npartitions=npartitions)
            else:
                # tif / tif stack
                images = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                        recursive=recursive, nplanes=nplanes, npartitions=npartitions)
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

            loader = SeriesLoader(self._sc, awsCredentialsOverride=self.awsCredentials)
            if inputFormat.lower() == 'stack':
                return loader.fromStack(dataPath, dims, ext=ext, dtype=dtype, blockSize=blockSize,
                                        startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)
            else:
                # tif / tif stack
                return loader.fromTif(dataPath, ext=ext, blockSize=blockSize,
                                      startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)

    def convertImagesToSeries(self, dataPath, outputDirPath, dims=None, inputFormat='stack', ext=None,
                              dtype='int16', blockSize="150M", blockSizeUnits="pixels", startIdx=None, stopIdx=None,
                              shuffle=True, overwrite=False, recursive=False, nplanes=None, npartitions=None,
                              renumber=False):
        """
        Write out Images data as Series data, saved in a flat binary format.

        The resulting Series data files may subsequently be read in using the loadSeries() method. The Series data
        object that results will be equivalent to that which would be generated by loadImagesAsSeries(). It is expected
        that loading Series data directly from the series flat binary format, using loadSeries(), will be faster than
        converting image data to a Series object through loadImagesAsSeries().

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

            The first dimension of the passed dims tuple should be the one that is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, z0), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)]. This is the opposite convention from that used by numpy,
            which by default has the fastest-changing dimension listed last (column-major convention). Thus, if loading
            a numpy array `ary`, where `ary.shape == (z, y, x)`, written to disk by `ary.tofile("myarray.stack")`, the
            corresponding dims parameter should be (x, y, z).
            If inputFormat is 'tif', the dims parameter (if any) will be ignored; data dimensions will instead
            be read out from the tif file headers.

        inputFormat: {'stack', 'tif'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data, while 'tif' indicates
            greyscale / luminance TIF images. Each page of a multipage tif file will be interpreted as a separate
            z-plane. For both stacks and tif stacks, separate files are interpreted as distinct time points, with
            ordering given by lexicographic sorting of file names.

        ext: string, optional, default None
            Extension required on data files to be loaded. By default will be "stack" if inputFormat=="stack", "tif" for
            inputFormat=='tif'.

        dtype: string or numpy dtype. optional, default 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputFormat is
            'tif', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            tif headers.

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
        """
        checkParams(inputFormat, ['stack', 'tif', 'tif-stack'])

        if inputFormat.lower() == 'stack' and not dims:
            raise ValueError("Dimensions ('dims' parameter) must be specified if loading from binary image stack" +
                             " ('stack' value for 'inputFormat' parameter)")

        if not overwrite:
            raiseErrorIfPathExists(outputDirPath, awsCredentialsOverride=self.awsCredentials)
            overwrite = True  # prevent additional downstream checks for this path

        if not ext:
            ext = DEFAULT_EXTENSIONS.get(inputFormat.lower(), None)

        if shuffle:
            from thunder.rdds.fileio.imagesloader import ImagesLoader
            loader = ImagesLoader(self._sc, awsCredentialsOverride=self.awsCredentials)
            if inputFormat.lower() == 'stack':
                images = loader.fromStack(dataPath, dims, ext=ext, dtype=dtype, startIdx=startIdx, stopIdx=stopIdx,
                                          recursive=recursive, nplanes=nplanes, npartitions=npartitions)
            else:
                # 'tif' or 'tif-stack'
                images = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                        recursive=recursive, nplanes=nplanes, npartitions=npartitions)
            if renumber:
                images = images.renumber()
            images.toBlocks(blockSize, units=blockSizeUnits).saveAsBinarySeries(outputDirPath, overwrite=overwrite)
        else:
            from thunder.rdds.fileio.seriesloader import SeriesLoader
            if nplanes is not None:
                raise NotImplementedError("nplanes is not supported with shuffle=False")
            if npartitions is not None:
                raise NotImplementedError("npartitions is not supported with shuffle=False")
            loader = SeriesLoader(self._sc, awsCredentialsOverride=self.awsCredentials)
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

    def loadExample(self, dataset):
        """
        Load a local example data set for testing analyses.

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset
        """

        import os

        path = os.path.dirname(os.path.realpath(__file__))
        # this path might actually be inside an .egg file (appears to happen with Spark 1.2)
        # check whether data/ directory actually exists on the filesystem, and if not, try
        # a hardcoded path that should work on ec2 clusters launched via the thunder-ec2 script
        if not os.path.isdir(os.path.join(path, 'data')):
            path = "/root/thunder/python/thunder/utils"

        if dataset == "iris":
            return self.loadSeries(os.path.join(path, 'data/iris/iris.bin'))
        elif dataset == "fish-series":
            return self.loadSeries(os.path.join(path, 'data/fish/bin/')).astype('float')
        elif dataset == "fish-images":
            return self.loadImages(os.path.join(path, 'data/fish/tif-stack'), inputFormat="tif")
        else:
            raise NotImplementedError("Dataset '%s' not known; should be one of 'iris', 'fish-series', 'fish-images'"
                                      % dataset)

    def loadExampleEC2(self, dataset):
        """
        Load an example data set from EC2.

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset

        params : Tuple or numpy array
            Parameters or metadata for dataset
        """

        import json
        from numpy import asarray

        if 'ec' not in self._sc.master:
            raise Exception("must be running on EC2 to load this example data sets")
        elif dataset == "zebrafish-optomotor-response":
            path = 'zebrafish.datasets/optomotor-response/1/'
            data = self.loadSeries("s3n://" + path + 'data/dat_plane*.txt', inputFormat='text', minPartitions=1000,
                                   nkeys=3)
            paramFile = self._sc.textFile("s3n://" + path + "params.json")
            params = json.loads(paramFile.first())
            modelFile = asarray(params['trials'])
            return data, modelFile
        else:
            raise NotImplementedError("dataset '%s' not availiable" % dataset)

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
        loader = SeriesLoader(self._sc, minPartitions=minPartitions, awsCredentialsOverride=self.awsCredentials)

        if inputFormat.lower() == 'mat':
            if varName is None:
                raise Exception('Must provide variable name for loading MAT files')
            data = loader.fromMatLocal(dataFilePath, varName, keyFilePath)
        else:
            data = loader.fromNpyLocal(dataFilePath, keyFilePath)

        return data

    def setAWSCredentials(self, awsAccessKeyId, awsSecretAccessKey):
        """Manually set AWS access credentials to be used by Thunder.

        This method is provided primarily for hosted environments that do not provide
        filesystem access (e.g. Databricks Cloud). Typically AWS credentials can be set
        and read from core-site.xml (for Hadoop input format readers, such as Series
        binary files), ~/.boto or other boto credential file locations, or the environment
        variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY. These credentials should
        be configured automatically in clusters launched by the thunder-ec2 script, and
        so this method should not have to be called.

        Parameters
        ----------
        awsAccessKeyId: string AWS public key, usually starts with "AKIA"
        awsSecretAccessKey: string AWS private key
        """
        from thunder.utils.common import AWSCredentials
        self.awsCredentials = AWSCredentials(awsAccessKeyId, awsSecretAccessKey)
        self._sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", awsAccessKeyId)
        self._sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", awsSecretAccessKey)


DEFAULT_EXTENSIONS = {
    "stack": "stack",
    "tif": "tif",
    "tif-stack": "tif",
    "png": "png"
}
