package thunder.util.io

import org.apache.spark.rdd.RDD
import java.io.FileOutputStream

/*** Class for writing an RDD to a flat binary file */

class BinaryWriter extends Writer with Serializable {

  def write(rdd: RDD[Double], fullFile: String) {
    val out = rdd.collect()
    val file = new FileOutputStream(fullFile ++ ".bin")
    val channel = file.getChannel
    val bbuf = java.nio.ByteBuffer.allocate(8*out.length)
    bbuf.asDoubleBuffer.put(java.nio.DoubleBuffer.wrap(out))

    while(bbuf.hasRemaining) {
      channel.write(bbuf)
    }

  }

}
