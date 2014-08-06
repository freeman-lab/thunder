package thunder.util.io

import org.apache.spark.rdd.RDD

/*** Class for writing an RDD to an image */

class ImageWriter extends Writer {

  def write(rdd: RDD[Double], fullFile: String) = {
    throw new NotImplementedError("image writing not yet implemented")
  }

}
