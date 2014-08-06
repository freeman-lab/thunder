package thunder.util

import org.apache.spark.streaming.dstream.DStream
import thunder.util.io.{TextWriter, BinaryWriter}

/** Object with methods for saving results from a DStream */

object SaveStreaming {

  def asTextWithKeys(data: DStream[(Int, Array[Double])], directory: String, fileName: Seq[String]) =
    new TextWriter().withKeys(data, directory, fileName)

  def asBinaryWithKeys(data: DStream[(Int, Array[Double])], directory: String, fileName: Seq[String]) =
    new BinaryWriter().withKeys(data, directory, fileName)

}