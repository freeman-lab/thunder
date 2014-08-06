package thunder.util.io

import org.apache.spark.rdd.RDD
import scala.math._
import org.apache.spark.streaming.dstream.DStream

/**
 * Functions for converting and querying keys
 */

object Keys {

  /**
   * Get maximum dimensions of data based on keys in an RDD
   *
   * @param data: RDD of data points as key value pairs
   */
  def getDims(data: RDD[(Array[Int], Array[Double])]): Array[Int] = {
    Range(0, data.first()._1.length).map(x => data.map(_._1(x)).reduce(max)).toArray
  }

  /** Inline function for converting subscript to linear indices */
  def subToIndLine(k: Array[Int], dimprod: Array[Int]): Int = {
    k.drop(1).zip(dimprod).map{case (x, y) => (x - 1) * y}.sum + k(0)
  }

  /** Inline function for converting linear to subscript indices */
  def indToSubLine(k: Int, dimprod: Array[(Int, Int)]): Array[Int] = {
    dimprod.map{case (x, y) => ((ceil(k.toDouble / y) - 1) % x + 1).toInt}
  }


  /** Convert subscript indices to linear indexing in an array
    *
    * @param data Array of Int Arrays with subscript indices
    * @param dims Dimensions for subscript indices
    * @return Array of Int with linear indices
    */
  def subToInd(data: Array[Array[Int]], dims: Array[Int]): Array[Int] = dims.length match {
    case 1 => data.map(x => x(0))
    case _ =>
      val dimprod = dims.scanLeft(1)(_*_).drop(1).dropRight(1)
      data.map(k => subToIndLine(k, dimprod))
  }


  /**
   * Convert subscript indexing to linear indexing in an RDD
   *
   * @param data RDD of data points as key value pairs
   * @param dims Dimensions for subscript indices
   * @return RDD of data points with linear indices as keys
   */
  def subToInd(data: RDD[(Array[Int], Array[Double])], dims: Array[Int]):
  RDD[(Int, Array[Double])] = dims.length match {
    case 1 => data.map{case (k, v) => (k(0), v)}
    case _ =>
      val dimprod = dims.scanLeft(1)(_*_).drop(1).dropRight(1)
      data.map{case (k, v) => (subToIndLine(k, dimprod), v)}
  }

  /**
   * Convert subscript indexing to linear indexing in a DStream
   *
   * @param data DStream of data points as key value pairs
   * @param dims Dimensions for subscript indices
   * @return DStream of data points with linear indices as keys
   */
  def subToIndStreaming(data: DStream[(Array[Int], Array[Double])], dims: Array[Int]):
  DStream[(Int, Array[Double])] = dims.length match {
    case 1 => data.map{case (k, v) => (k(0), v)}
    case _ =>
      val dimprod = dims.scanLeft(1)(_*_).drop(1).dropRight(1)
      data.map{case (k, v) => (subToIndLine(k, dimprod), v)}
  }


  /**
   * Convert linear indexing to subscript indexing in an array
   *
   * @param data Array of Int with linear indices
   * @param dims Dimensions for subscript indices
   * @return Array of Int Array with subscript indices
   */
  def indToSub(data: Array[Int], dims: Array[Int]):
  Array[Array[Int]] = dims.length match {
    case 1 => data.map(x => Array(x))
    case _ =>
      val dimprod = dims.zip(Array.concat(Array(1), dims.map(x => Range(1, x + 1).reduce(_*_)).dropRight(1)))
      data.map(x => indToSubLine(x, dimprod))
  }

  /**
   * Convert linear indexing to subscript indexing in an RDD
   *
   * @param data RDD of data points as key value pairs
   * @param dims Dimensions for subscript indices
   * @return RDD of data points with subscript indices as keys
   */
  def indToSub(data: RDD[(Int, Array[Double])], dims: Array[Int]):
  RDD[(Array[Int], Array[Double])] = dims.length match {
    case 1 => data.map{case (k, v) => (Array(k), v)}
    case _ =>
      val dimprod = dims.zip(Array.concat(Array(1), dims.map(x => Range(1, x + 1).reduce(_*_)).dropRight(1)))
      data.map{case (k, v) => (indToSubLine(k, dimprod), v)}
  }

  /**
   * Convert linear indexing to subscript indexing in a DStream
   *
   * @param data DStream of data points as key value pairs
   * @param dims Dimensions for subscript indices
   * @return DStream of data points with subscript indices as keys
   */
  def indToSubStreaming(data: DStream[(Int, Array[Double])], dims: Array[Int]):
  DStream[(Array[Int], Array[Double])] = dims.length match {
    case 1 => data.map{case (k, v) => (Array(k), v)}
    case _ =>
      val dimprod = dims.zip(Array.concat(Array(1), dims.map(x => Range(1, x + 1).reduce(_*_)).dropRight(1)))
      data.map{case (k, v) => (indToSubLine(k, dimprod), v)}
  }


}
