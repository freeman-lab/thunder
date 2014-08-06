package thunder.util

import org.apache.spark.util.StatCounter


/**
 * A class for tracking multiple sets of running statistics via an array
 * of StatCounters, and extracting the statistics (count, mean, variance, etc.)
 * as arrays. Support for merging two StatCounterArrays, by merging
 * the underlying StatCounters element-wise.
 */
class StatCounterArray(values: Array[TraversableOnce[Double]], n: Int) extends Serializable {

  val num: Int = n
  var counters: Array[StatCounter] = Array.fill(n)(new StatCounter())

  merge(values)

  /** Initialize an array of empty StatCounters with no values. */
  def this(n: Int) = this(Array.fill(n)(Nil), n)

  /** Initialize a StatCounterArray with the default of 2 StatCounters. */
  def this() = this(2)

  /** Merge an Array of values into a StatCounterArray, supporting different input types */
  def merge(values: Array[Double]): StatCounterArray = {
    if (this.num != values.length) {
      throw new IllegalArgumentException("cannot merge %d values to %d StatCounters".format(values.length, this.num))
    }
    counters.zip(values).foreach{case (counter, data) => counter.merge(data)}
    this
  }

  def merge(values: Array[Array[Double]]): StatCounterArray = {
    if (this.num != values.length) {
      throw new IllegalArgumentException("cannot merge %d values to %d StatCounters".format(values.length, this.num))
    }
    counters.zip(values).foreach{case (counter, data) => counter.merge(data)}
    this
  }

  def merge(values: Array[TraversableOnce[Double]]): StatCounterArray = {
    if (this.num != values.length) {
      throw new IllegalArgumentException("cannot merge %d values to %d StatCounters".format(values.length, this.num))
    }
    counters.zip(values).foreach{case (counter, data) => counter.merge(data)}
    this
  }

  /** Merge two StatCounterArrays together */
  def merge(other: StatCounterArray): StatCounterArray = {
    if (this.num != other.num) {
      throw new IllegalArgumentException("cannot merge StatCounterArray sizes %d and %d".format(this.num, other.num))
    }
    counters.zip(other.counters).foreach{case (counter1, counter2) => counter1.merge(counter2)}
    this
  }

  /** Return Arrays of statistics, see StatCounter for details */
  def count: Array[Long] = counters.map(c => c.count)

  def mean: Array[Double] = counters.map(c => c.mean)

  def sum: Array[Double] = counters.map(c => c.sum)

  def variance: Array[Double] = counters.map(c => c.variance)

  def sampleVariance: Array[Double] = counters.map(c => c.sampleVariance)

  def stdev: Array[Double] = counters.map(c => c.stdev)

  def sampleStdev: Array[Double] = counters.map(c => c.sampleStdev)

  def combinedVariance: Double = {
    counters.map(c => c.variance * c.count).filter(x => !x.isNaN).sum / this.count.sum
  }

  override def toString: String = {
    counters.map(c => c.toString()).mkString("\n")
  }

}

object StatCounterArray {

  /** Build a StatCounter from a nested Array. */
  def apply(values: Array[TraversableOnce[Double]]) = new StatCounterArray(values, values.length)

  def apply(values: Array[Array[Double]]) = new StatCounterArray(values.map(_.toTraversable), values.length)

  /** Build a StatCounter from a flat Array. */
  def apply(values: Array[Double]) = new StatCounterArray(values.map(x => Array(x).toTraversable), values.length)

}