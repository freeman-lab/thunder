package thunder

import org.scalatest.Assertions._

/**
 * Useful tools for testing analysis algorithms, borrowed with modification
 * from MLlib testing suite
 */
object TestUtils {

  def prettyPrint(point: Array[Double]): String = point.mkString("(", ", ", ")")

  def prettyPrint(points: Array[Array[Double]]): String = {
    points.map(prettyPrint).mkString("(", "; ", ")")
  }

  // L1 distance between two points
  def distance1(v1: Array[Double], v2: Array[Double]): Double = {
    v1.zip(v2).map{ case (a, b) => math.abs(a-b) }.max
  }

  // Assert that two vectors are equal within tolerance EPSILON
  def assertEqual(v1: Double, v2: Double, epsilon: Double) {
    def errorMessage = v1.toString + " did not equal " + v2.toString
    assert(math.abs(v1-v2) <= epsilon, errorMessage)
  }

  // Assert that two vectors are equal within tolerance EPSILON
  def assertEqual(v1: Array[Double], v2: Array[Double], epsilon: Double) {
    def errorMessage = prettyPrint(v1) + " did not equal " + prettyPrint(v2)
    assert(v1.length == v2.length, errorMessage)
    assert(distance1(v1, v2) <= epsilon, errorMessage)
  }

  // Assert that two sets of points are equal, within EPSILON tolerance
  def assertSetsEqual(set1: Array[Array[Double]], set2: Array[Array[Double]], epsilon: Double) {
    def errorMessage = prettyPrint(set1) + " did not equal " + prettyPrint(set2)
    assert(set1.length == set2.length, errorMessage)
    for (v <- set1) {
      val closestDistance = set2.map(w => distance1(v, w)).min
      if (closestDistance > epsilon) {
        fail(errorMessage)
      }
    }
    for (v <- set2) {
      val closestDistance = set1.map(w => distance1(v, w)).min
      if (closestDistance > epsilon) {
        fail(errorMessage)
      }
    }
  }
}