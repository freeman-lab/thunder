package thunder.util.io

import org.apache.spark.rdd.RDD
import java.io.File
import thunder.util.Save

/*** Class for writing an RDD to a text file */

class TextWriter extends Writer with Serializable {

  def write(rdd: RDD[Double], fullFile: String) {
    val out = rdd.collect()
    printToFile(new File(fullFile ++ ".txt"))(p => {
      out.foreach(x => p.println("%.6f".format(x)))
    })
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try {
      op(p)
    } finally {
      p.close()
    }
  }


}

