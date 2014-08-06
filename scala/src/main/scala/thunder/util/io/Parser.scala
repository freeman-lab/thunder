package thunder.util.io

import java.nio.{ByteBuffer, ByteOrder}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors


/**
 * Class for loading lines of a data file
 * supporting a variety of formats
 *
 * @param nKeys Number of integer keys per data point
 * @param format Byte encoding
 */
case class Parser(nKeys: Int = 0, format: String = "Int") {

  /**
   * Convert an Array[Byte] to Array[Double]
   * using a Java ByteBuffer, where "format"
   * specifies the byte encoding scheme (and which
   * ByteBuffer subclass to use)
   */
  def convertBytes(v: Array[Byte]): Array[Double] = {

    format match {
      case "short" => {
        val buffer = ByteBuffer.wrap(v).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val intArray = new Array[Int](buffer.remaining())
        var t = 0
        while (buffer.remaining() > 0) {
          intArray(t) = buffer.get()
          t += 1
        }
        intArray.map(_.toDouble)
      }
      case "int" => {
        val buffer = ByteBuffer.wrap(v).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
        val intArray = new Array[Int](buffer.remaining())
        var t = 0
        while (buffer.remaining() > 0) {
          intArray(t) = buffer.get()
          t += 1
        }
        intArray.map(_.toDouble)
      }
      case "double" => {
        val buffer = ByteBuffer.wrap(v).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
        val DoubleArray = new Array[Double](buffer.remaining())
        var t = 0
        while (buffer.remaining() > 0) {
          DoubleArray(t) = buffer.get()
          t += 1
        }
        DoubleArray
      }
    }
  }

  /** nKeys per string, but only keep the values */
  def get(line: String): Array[Double] = {
    val parts = line.split(' ')
    val value = parts.slice(nKeys, parts.length).map(_.toDouble)
    value
  }

  /** nKeys per Array[Int], but only keep the values */
  def get(line: Array[Int]): Array[Double] = {
    val value = line.slice(nKeys, line.length).map(_.toDouble)
    value
  }

  /** nKeys per Array[Byte], but only keep the values */
  def get(line: Array[Byte]): Array[Double] = {
    val value = convertBytes(line).slice(nKeys, line.length)
    value
  }

  /** nKeys per string, keep both */
  def getWithKeys(line: String): (Array[Int], Array[Double]) = {
    val parts = line.split(' ')
    val key = parts.slice(0,nKeys).map(_.toDouble.toInt)
    val value = parts.slice(nKeys, parts.length).map(_.toDouble)
    (key, value)
  }

  /** nKeys per Array[Int], keep both */
  def getWithKeys(line: Array[Int]): (Array[Int], Array[Double]) = {
    val key = line.slice(0,nKeys)
    val value = line.slice(nKeys, line.length).map(_.toDouble)
    (key, value)
  }

  /** nKeys per Array[Byte], keep both */
  def getWithKeys(line: Array[Byte]): (Array[Int], Array[Double]) = {
    val parts = convertBytes(line)
    val key = parts.slice(0,nKeys).map(_.toInt)
    val value = parts.slice(nKeys, line.length)
    (key, value)
  }

  /** Single label per string, separated by a comma */
  def getWithLabels(line: String): LabeledPoint = {
    val parts = line.split(',')
    val label = parts(0).toDouble
    val features = Vectors.dense(parts(1).trim().split(' ').map(_.toDouble))
    LabeledPoint(label, features)
  }

  /** Single label per Array[Int], first value is the label */
  def getWithLabels(line: Array[Int]): LabeledPoint = {
    val label = line(0).toDouble
    val features = Vectors.dense(line.slice(1, line.length).map(_.toDouble))
    LabeledPoint(label, features)
  }

  /** Single label per Array[Byte], first value is the label */
  def getWithLabels(line: Array[Byte]): LabeledPoint = {
    val tmp = convertBytes(line)
    val label = tmp(0).toInt
    val features = Vectors.dense(tmp.slice(1, line.length))
    LabeledPoint(label, features)
  }

}