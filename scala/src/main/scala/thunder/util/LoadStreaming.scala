package thunder.util

import thunder.util.io.Parser
import thunder.util.io.Keys
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.mllib.regression.LabeledPoint
import thunder.util.io.hadoop.FixedLengthBinaryInputFormat
import org.apache.hadoop.io.{BytesWritable, LongWritable}


object LoadStreaming {

  /**
   * Load streaming data from text files with format
   * <t1> <t2> ...
   * Where <t1> <t2> are the data values (Double)
   *
   * @param ssc StreamingContext
   * @param dir Directory to the input data files
   * @return A DStream of RDDs, each a collection of Array[Double] records
   */
  def fromText (ssc: StreamingContext,
                         dir: String): DStream[Array[Double]] = {
    val parser = new Parser(0)
    ssc.textFileStream(dir).map(parser.get)
  }

  /**
   * Load keyed streaming data from text files with format
   * <k1> <k2> ... <t1> <t2> ...
   * where <k1> <k2> ... are keys (Int) and <t1> <t2> ... are the data values (Double)
   * If multiple keys are provided (e.g. x, y, z), they are converted to linear indexing
   *
   * @param ssc StreamingContext
   * @param dir Directory to the input data files
   * @param nKeys Number of keys per data point
   * @param dims Dimensions for converting subscript to linear indices
   * @return A DStream of RDDs, each a collection of (Int, Array[Double]) records
   */
  def fromTextWithKeys (ssc: StreamingContext,
                                 dir: String,
                                 nKeys: Int = 1,
                                 dims: Array[Int] = Array(1)): DStream[(Int, Array[Double])] = {
    val parser = new Parser(nKeys)
    Keys.subToIndStreaming(ssc.textFileStream(dir).map(parser.getWithKeys), dims)
  }

  /**
   * Load labeled streaming data from text files with format
   * <label>, <t1> <t2> ...
   * where <label> is the label and <t1> <t2> ... are the data values (Double)
   *
   * @param ssc StreamingContext
   * @param dir Directory to the input data files
   * @return A DStream of RDDs, each a collection of LabeledPoints
   */
  def fromTextWithLabels (ssc: StreamingContext,
                                dir: String): DStream[LabeledPoint] = {
    val parser = new Parser()
    ssc.textFileStream(dir).map(parser.getWithLabels)
  }

  /**
   * Load streaming data from flat binary files, assuming each record is a set of numbers
   * with the specified numerical format (see ByteBuffer) and the number of
   * bytes per record is constant (see FixedLengthBinaryInputFormat)
   *
   * @param ssc StreamingContext
   * @param dir Directory to the input data files
   * @param format Numerical format for conversion from binary
   * @return A DStream of data, each a collection of Array[Double]
   */
  def fromBinary (ssc: StreamingContext,
                dir: String,
                format: String = "int"): DStream[Array[Double]] = {
    val parser = new Parser(0, format)
    val lines = ssc.fileStream[LongWritable, BytesWritable, FixedLengthBinaryInputFormat](dir)
    lines.map{ case (k, v) => v.getBytes}.map(parser.get)
  }

  /**
   * Load streaming data from flat binary files, assuming each record is a set of numbers
   * with the specified numerical format (see ByteBuffer) and the number of
   * bytes per record is constant (see FixedLengthBinaryInputFormat). The first
   * nKeys numbers in each record are keys, and the rest are values.
   *
   * @param ssc StreamingContext
   * @param dir Directory to the input data files
   * @param nKeys Number of keys per record
   * @param format Numerical format for conversion from binary
   * @return A DStream of data, each a tuple with two elements: the key (an Int)
   *         and the the data values (an array of Double)
   */
  def fromBinaryWithKeys (ssc: StreamingContext,
                  dir: String,
                  nKeys: Int = 3,
                  dims: Array[Int] = Array(1),
                  format: String = "int"): DStream[(Int, Array[Double])] = {
    val parser = new Parser(nKeys, format)
    val lines = ssc.fileStream[LongWritable, BytesWritable, FixedLengthBinaryInputFormat](dir)
    Keys.subToIndStreaming(lines.map{ case (k, v) => v.getBytes}.map(parser.getWithKeys), dims)
  }

}
