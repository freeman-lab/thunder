package thunder.util

import scala.io.Source

/*** Load parameters from a file containing key value pairs
  * specified as:
  *
  * key: value
  *
  * One pair should be on each line. If multiple values are provided
  * for a given key, they will be collected into a single string separated
  * by line breaks. See ThunderParam.
  *
  * */

object LoadParam {

  def fromText(file: String): ThunderParam = {
    val param = new ThunderParam()
    val lines = Source.fromFile(file).getLines()
    lines.foreach{l =>
      val parts = l.split(": ")
      val key = parts(0)
      val value = parts(1)
      param.set(key, value)
    }
    param
  }

}
