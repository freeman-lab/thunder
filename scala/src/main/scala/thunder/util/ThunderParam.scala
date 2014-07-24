package thunder.util

import scala.collection.mutable.HashMap

/*** Class for storing parameters related to an analysis in Thunder */

class ThunderParam {

  private val settings = new HashMap[String, String]()

  /** Set a parameter. */
  def set(key: String, value: String): ThunderParam = {
    if (key == null) {
      throw new NullPointerException("null key")
    }
    if (value == null) {
      throw new NullPointerException("null value")
    }
    val current = settings.get(key).getOrElse("")
    if (current == "") {
      settings(key) = value
    } else {
      settings(key) = current + "\n" + value
    }
    this
  }

  /** Get any parameter, throw a NoSuchElementException if it's not set */
  def get(key: String): String = {
    settings.getOrElse(key, throw new NoSuchElementException(key))
  }

  def getDims: Array[Int] = {
    settings.get("dims").get.drop(1).dropRight(1).split(",").map(_.trim.toDouble).map(_.toInt)
  }

  def getBinKeys: Array[Array[Int]] = {
    settings.get("binKeys").get.split("\n").map(_.drop(1).dropRight(1).split(",").map(_.trim.toDouble).map(_.toInt))
  }

  def getBinValues: Array[Array[Double]] = {
    settings.get("binValues").get.split("\n").map(_.drop(1).dropRight(1).split(",").map(_.trim.toDouble))
  }

  def getBinName: Array[String] = {
    settings.get("binNames").get.split("\n")
  }

  def toDebugString: String = {
    settings.toArray.map{case (k, v) => k + "=" + v}.mkString("\n")
  }

}

