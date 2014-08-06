package thunder.util.io

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import java.io.File
import org.apache.spark.streaming.dstream.DStream
import java.util.Calendar

/**
 * Generic writer class for writing the contents of an RDD or DStream to disk.
 * Separate methods for handing RDDs with or without keys.
 *
 * This class should be extended by defining a write method specific
 * to a particular output type (see e.g. TextWriter, BinaryWriter).
 *
 */
abstract class Writer {

  def write(rdd: RDD[Double], fullFile: String)

  def withoutKeys(data: RDD[Array[Double]], directory: String, fileName: Seq[String]) {
    val n = data.first().size
    for (i <- 0 until n) {
      write(data.map(x => x(i)), directory ++ File.separator ++ fileName(i))
    }
  }

  def withKeys(data: RDD[(Array[Int], Array[Double])], directory: String, fileName: Seq[String]) {
    val dims = Keys.getDims(data)
    val sorted = Keys.subToInd(data, dims).sortByKey().values
    val n = sorted.first().size
    for (i <- 0 until n) {
      write(sorted.map(x => x(i)), directory ++ File.separator ++ fileName(i))
    }
  }

  def withKeys(data: DStream[(Int, Array[Double])], directory: String, fileName: Seq[String]) {
    data.foreachRDD{rdd =>
      val sorted = rdd.sortByKey().values
      val n = sorted.first().size
      val dateString = Calendar.getInstance().getTime.toString.replace(" ", "-").replace(":", "-")
      for (i <- 0 until n) {
        write(sorted.map(x => x(i)), directory ++ File.separator ++ fileName(i) ++ "-" ++ dateString)
      }
    }
  }

}