package thunder.util

import org.apache.spark.rdd.RDD
import thunder.util.io.{BinaryWriter, TextWriter}

/** Object with methods for saving results from an RDD */

object Save {

  def asText(data: RDD[Array[Double]], directory: String, fileName: Seq[String]) =
    new TextWriter().withoutKeys(data, directory, fileName)

  def asTextWithKeys(data: RDD[(Array[Int], Array[Double])], directory: String, fileName: Seq[String]) =
    new TextWriter().withKeys(data, directory, fileName)

  def asBinary(data: RDD[Array[Double]], directory: String, fileName: Seq[String]) =
    new BinaryWriter().withoutKeys(data, directory, fileName)

  def asBinaryWithKeys(data: RDD[(Array[Int], Array[Double])], directory: String, fileName: Seq[String]) =
    new BinaryWriter().withKeys(data, directory, fileName)

}