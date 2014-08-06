
package thunder.streaming

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.SparkContext._
import org.apache.spark.util.StatCounter
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream

import thunder.util.{SaveStreaming, StatCounterArray, LoadParam, LoadStreaming}
import thunder.util.io.Keys
import scala.Some


/**
 * Class for keeping track of a (StatCounter, StatCounterArray) pair, and computing
 * summary statistics derived from them. Typically, the StatCounter will compute
 * a global running statistic, and the StatCounterArray will compute running
 * statistics within each of several bins.
 *
 */
case class BinnedStats (var counterTotal: StatCounter, var counterBins: StatCounterArray) {

  def r2: Double = {
    1 - counterBins.combinedVariance / counterTotal.variance
  }

  // TODO minimum should only be for data where count is not 0
  def weightedMean(featureValues: Array[Double]): Double = {
    val means = counterBins.mean
    val pos = means.map(x => x - means.min)
    val weights = pos.map(x => x / pos.sum)
    weights.zip(featureValues).map{case (x,y) => x * y}.sum
  }

}


/**
 * Stateful binned statistics
 */
class StatefulBinnedStats (
  var featureKeys: Array[Int],
  var nFeatures: Int)
  extends Serializable with Logging
{

  def this() = this(Array(0), 1)

  /** Set which indices that correspond to features. */
  def setFeatureKeys(featureKeys: Array[Int]): StatefulBinnedStats = {
    this.featureKeys = featureKeys
    this
  }

  /** Set the values associated with the to features. */
  def setFeatureCount(nFeatures: Int): StatefulBinnedStats = {
    this.nFeatures = nFeatures
    this
  }

  /** State updating function that updates the statistics for each key and bin. */
  val runningBinnedStats = (input: Seq[Array[Double]],
                            state: Option[BinnedStats],
                            features: Array[Double]) => {

    val updatedState = state.getOrElse(BinnedStats(new StatCounter(), new StatCounterArray(nFeatures)))

    val values = input.foldLeft(Array[Double]()) { (acc, i) => acc ++ i}
    val currentCount = values.size
    val n = features.size

    if ((currentCount != 0) && (n != 0)) {

      // group new data by the features
      val pairs = features.zip(values)
      val grouped = pairs.groupBy{case (k,v) => k}

      // get data from each bin, ignoring the 0 bin
      val binnedData = Range(1,nFeatures+1).map{ ind => if (grouped.contains(ind)) {
        grouped(ind).map{ case (k,v) => v}
        } else {
          Array[Double]()
        }
      }.toArray

      // get all data, ignoring the 0 bin
      val all = pairs.filter{case (k,v) => k != 0}.map{case (k,v) => v}

      // update the combined stat counter
      updatedState.counterTotal.merge(all)

      // update the binned stat counter array
      updatedState.counterBins.merge(binnedData)

    }
    Some(updatedState)
  }

  def runStreaming(data: DStream[(Int, Array[Double])]): DStream[(Int, BinnedStats)] = {

    var features = Array[Double]()

    // extract the bin labels
    data.filter{case (k, _) => featureKeys.contains(k)}.foreachRDD{rdd =>
      val batchFeatures = rdd.values.collect().flatten
      features = batchFeatures.size match {
        case 0 => Array[Double]()
        case _ => batchFeatures
      }
      println(features.mkString(" "))
    }

    // update the stats for each key
    data.filter{case (k, _) => !featureKeys.contains(k)}.updateStateByKey{
      (x, y) => runningBinnedStats(x, y, features)
    }

  }

}

/**
 * Top-level methods for calling Stateful Binned Stats.
 */
object StatefulBinnedStats {

  /**
   * Compute running statistics on keyed data points in bins.
   * For each key, statistics are computed within each of several bins
   * specified by auxiliary data passed as a special key.
   * We assume that in each batch of streaming data we receive
   * an array of doubles for each data key, and an array of integer indices
   * for the bin key. We use a StatCounterArray on
   * each key to update the statistics within each bin.
   *
   * @param input DStream of (Int, Array[Double]) keyed data point
   * @return DStream of (Int, BinnedStats) with statistics
   */
  def trainStreaming(input: DStream[(Int, Array[Double])],
                     featureKeys: Array[Int],
                     featureCount: Int): DStream[(Int, BinnedStats)] =
  {
    new StatefulBinnedStats().setFeatureKeys(featureKeys).setFeatureCount(featureCount).runStreaming(input)
  }

  def main(args: Array[String]) {
    if (args.length != 5) {
      System.err.println(
        "Usage: StatefulBinnedStats <master> <directory> <batchTime> <outputDirectory> <paramFile>")
      System.exit(1)
    }

    val (master, directory, batchTime, outputDirectory, paramFile) = (
      args(0), args(1), args(2).toLong, args(3), args(4))

    val conf = new SparkConf().setMaster(master).setAppName("StatefulStats")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
        .setJars(List("target/scala-2.10/thunder_2.10-0.1.0.jar"))
        .set("spark.executor.memory", "100G")
        .set("spark.default.parallelism", "100")
    }

    /** Create Streaming Context */
    val ssc = new StreamingContext(conf, Seconds(batchTime))
    ssc.checkpoint(System.getenv("CHECKPOINT"))

    /** Load analysis parameters */
    val params = LoadParam.fromText(paramFile)

    /** Get feature keys with linear indexing */
    val binKeys = Keys.subToInd(params.getBinKeys, params.getDims)

    /** Get values for the features */
    val binValues = params.getBinValues

    /** Get names for the bins */
    val binName = params.getBinName

    /** Load streaming data */
    val data = LoadStreaming.fromTextWithKeys(ssc, directory, params.getDims.size, params.getDims)

    /** Train the models */
    val state = StatefulBinnedStats.trainStreaming(data, binKeys, binValues(0).length)

    /** Print results (for testing) */
    val result = state.mapValues(x => Array(x.r2))

    /** Collect output */
    SaveStreaming.asTextWithKeys(result, outputDirectory, Seq("r2-" + binName(0)))

    ssc.start()
    ssc.awaitTermination()
  }

}
