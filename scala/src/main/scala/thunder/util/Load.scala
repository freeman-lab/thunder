/** Utilities for loading and parsing data */

package thunder.util

import thunder.util.io.Parser
import thunder.util.io.PreProcessor
import thunder.util.io.hadoop.FixedLengthBinaryInputFormat
import org.apache.hadoop.conf.Configuration
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.hadoop.io.{BytesWritable, LongWritable}
import scala.reflect.{classTag, ClassTag}


object Load {

  /**
   * Load data from a text file with format
   * <t1> <t2> ...
   * where <t1> <t2> ... are the data values
   *
   * @param sc SparkContext
   * @param dir Directory to the input data files
   * @param preProcessMethod Method for pre processing data (default = "raw")
   * @return An RDD of data with values, RDD[(Array[Double])]
   */
  def fromText(sc: SparkContext,
                       dir: String,
                       preProcessMethod: String = "raw"): RDD[Array[Double]] = {
    val parser = new Parser(0)

    if (preProcessMethod != "raw") {
      val processor = new PreProcessor(preProcessMethod)
      sc.textFile(dir).map(parser.get).map(processor.get)
    } else {
      sc.textFile(dir).map(parser.get)
    }
  }

  /**
   * Load keyed data from a text file with format
   * <k1> <k2> ... <t1> <t2> ...
   * where <k1> <k2> ... are keys and <t1> <t2> ... are the data values
   *
   * @param sc SparkContext
   * @param dir Directory to the input data files
   * @param preProcessMethod Method for pre processing data (default = "raw")
   * @param nKeys Number of keys per data point (default = 3)
   * @return An RDD of data with keys and values, RDD[(Array[Int], Array[Double])]
   */
  def fromTextWithKeys(sc: SparkContext,
               dir: String,
               preProcessMethod: String = "raw",
               nKeys: Int = 3): RDD[(Array[Int], Array[Double])] = {
    val parser = new Parser(nKeys)

    if (preProcessMethod != "raw") {
      val processor = new PreProcessor(preProcessMethod)
      sc.textFile(dir).map(parser.getWithKeys).mapValues(processor.get)
    } else {
      sc.textFile(dir).map(parser.getWithKeys)
    }
  }

  /**
   * Load data from a flat binary file, assuming each record is a set of numbers
   * with the specified numerical format (see ByteBuffer), and the number of
   * bytes per record is constant (see FixedLengthBinaryInputFormat)
   *
   * @param sc SparkContext
   * @param dir Directory to the input data files
   * @param preProcessMethod Method for pre processing data (default = "raw")
   * @param format Numerical format for conversion from binary
   * @return An RDD of data with values, RDD[(Array[Double])]
   */
  def fromBinary(sc: SparkContext,
               dir: String,
               recordLength: Int,
               preProcessMethod: String = "raw",
               format: String = "int"): RDD[Array[Double]] = {
    val parser = new Parser(0, format)

    val conf = new Configuration()
    val lines = sc.newAPIHadoopFile(
      dir, classOf[FixedLengthBinaryInputFormat], classOf[LongWritable], classOf[BytesWritable], conf)
    val data = lines.map{ case (k, v) => v.getBytes}

    if (preProcessMethod != "raw") {
      val processor = new PreProcessor(preProcessMethod)
      data.map(parser.get).map(processor.get)
    } else {
      data.map(parser.get)
    }
  }

  /**
   * Load data from a flat binary file, assuming each record is a set of numbers
   * with the specified numerical format (see ByteBuffer) and the number of
   * bytes per record is constant (see FixedLengthBinaryInputFormat). The first
   * nKeys numbers in each record are the keys and the remaining numbers are the values.
   *
   * @param sc SparkContext
   * @param dir Directory to the input data files
   * @param preProcessMethod Method for pre processing data (default = "raw")
   * @param nKeys Method for pre processing data (default = "raw")
   * @param format Numerical format for conversion from binary
   * @return An RDD of data with values, RDD[(Array[Double])]
   */
  def fromBinaryWithKeys(sc: SparkContext,
                 dir: String,
                 recordLength: Int,
                 preProcessMethod: String = "raw",
                 nKeys: Int = 3,
                 format: String = "int"): RDD[(Array[Int], Array[Double])] = {

    val parser = new Parser(nKeys, format)
    val conf = new Configuration()
    conf.set("recordLength", "204")
    val lines = sc.newAPIHadoopFile(
      dir, classOf[FixedLengthBinaryInputFormat], classOf[LongWritable], classOf[BytesWritable], conf)
    val data = lines.map{ case (k, v) => v.getBytes}

    if (preProcessMethod != "raw") {
      val processor = new PreProcessor(preProcessMethod)
      data.map(parser.getWithKeys).mapValues(processor.get)
    } else {
      data.map(parser.getWithKeys)
    }
  }

 }


