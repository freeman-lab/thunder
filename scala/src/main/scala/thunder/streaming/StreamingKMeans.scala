package thunder.streaming

import org.apache.spark.{SparkConf, Logging}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.streaming.StreamingKMeansModel

import scala.util.Random.nextDouble
import scala.util.Random.nextGaussian

import thunder.util.LoadStreaming

/**
 * K-means clustering on streaming data with support for
 * sequential and forgetful algorithms.
 *
 * The underlying assumption is that all streaming data points
 * belong to one of several clusters, and we want to
 * learn the identity of those clusters (the "KMeans Model")
 * as new data arrive. All records MUST have the same dimensionality.
 *
 * For sequential algorithms, we update the underlying
 * cluster identities once for each batch of data, and keep
 * a count of the number of data points per cluster.
 * The number of data points per batch can be arbitrary.
 * This implementation is based on the sequential k-means algorithm
 * (see https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm)
 * except that each update is based on a batch of data rather than
 * a single data point. It is also similar to the offline mini-batch
 * algorithm (see http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)
 * except that batches arrive over time, rather than through sampling.
 *
 * For forgetful algorithms, each new batch of data is weighted in
 * its contribution so that more recent data is weighted more strongly
 * (see https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm).
 * The weighting is per batch (i.e. per time window), rather than per data point,
 * so for meaningful interpretation, the number of data points per batch
 * should be approximately constant.
 *
 */
class StreamingKMeans (
  var k: Int,
  var d: Int,
  var a: Double,
  var maxIterations: Int,
  var initializationMode: String)
  extends Serializable with Logging
{

  private type ClusterCentersAndCounts = Array[(Array[Double], Int)]

  /** Construct a StreamingKMeans object with default parameters */
  def this() = this(2, 5, 1.0, 1, "gauss")

  /** Set the number of clusters to create (k). Default: 2. */
  def setK(k: Int): StreamingKMeans = {
    this.k = k
    this
  }

  /** Set the dimensionality of the data (d). Default: 5
    * TODO: if possible, set this automatically based on first data point
    */
  def setD(d: Int): StreamingKMeans = {
    this.d = d
    this
  }

  /**
   * Set the parameter alpha to determine the update rule.
   * If alpha = 1, perform seqeutnail or "mini batch" KMeans, treating all
   * data equivalently. If alpha < 1, perform forgetful KMeans,
   * which uses a constant to weight old data
   * less strongly (with exponential weighting), e.g. 0.9 will
   * favor only recent data, whereas 0.1 will update slowly.
   * Weighting over time is per batch, so this algorithm implicitly
   * assumes an approximately constant number of data points per batch
   * Default: 1 (sequential)
   */
  def setAlpha(a: Double): StreamingKMeans = {
    this.a = a
    this
  }

  // TODO: characterize the effect of max iterations in forgetful version
  /** Set the maximum number of iterations per batch of data. */
  def setMaxIterations(maxIterations: Int): StreamingKMeans = {
    this.maxIterations = maxIterations
    this
  }

  /**
   * Set the initialization algorithm. Unlike batch KMeans, we
   * initialize randomly before we have seen any data. Options are "gauss"
   * for random Gaussian centers, and "pos" for random positive uniform centers.
   * Default: gauss
   */
  def setInitializationMode(initializationMode: String): StreamingKMeans = {
    if (initializationMode != "gauss" && initializationMode != "pos") {
      throw new IllegalArgumentException("Invalid initialization mode: " + initializationMode)
    }
    this.initializationMode = initializationMode
    this
  }


  /** Initialize random points for KMeans clustering */
  def initRandom(): StreamingKMeansModel = {

    val clusters = new Array[(Array[Double], Int)](k)
    for (ik <- 0 until k) {
      clusters(ik) = initializationMode match {
        case "gauss" => (Array.fill(d)(nextGaussian()), 0)
        case "pos" => (Array.fill(d)(nextDouble()), 0)
      }
    }
    new StreamingKMeansModel(clusters.map(_._1).map(x => Vectors.dense(x)), clusters.map(_._2))
  }

  // TODO: stop iterating if clusters have converged
  /** Update KMeans clusters by doing training passes over an RDD */
  def update(data: RDD[Vector], model: StreamingKMeansModel): StreamingKMeansModel = {

    val centers = model.clusterCenters
    val counts = model.clusterCounts

    // do iterative KMeans updates on a batch of data
    for (i <- Range(0, maxIterations)) {
      // find nearest cluster to each point
      val closest = data.map(point => (model.predict(point), (point, 1)))

      // get sums and counts for updating each cluster
      val pointStats = closest.reduceByKey{
        case ((x1, y1), (x2, y2)) => (Vectors.dense(x1.toArray.zip(x2.toArray).map{case (x, y) => x + y}), y1 + y2)}
      val newPoints = pointStats.map{
        pair => (pair._1, (pair._2._1, pair._2._2))}.collectAsMap()

      a match {
        case 1 => for (newP <- newPoints) {
          // remove previous count scaling
          centers(newP._1) = Vectors.dense(centers(newP._1).toArray.map(x => x * counts(newP._1)))
          // update sums
          centers(newP._1) = Vectors.dense(centers(newP._1).toArray.zip(newP._2._1.toArray).map{case (x, y) => x + y})
          // update counts
          counts(newP._1) += newP._2._2
          // rescale to compute new means (of both old and new points)
          centers(newP._1) = Vectors.dense(centers(newP._1).toArray.map(x => x / counts(newP._1)))
        }
        case _ => for (newP <- newPoints) {
          // update centers with forgetting factor a
          centers(newP._1) = Vectors.dense(centers(newP._1).toArray.zip(newP._2._1.toArray.map(x => x / newP._2._2)).map{
            case (x, y) => x + a * (y - x)})
        }
      }

      val model.clusterCenters = centers
    }

    // log the cluster centers
    centers.zip(Range(0, centers.length)).foreach{
      case (x, ix) => logInfo("Cluster center " + ix.toString + ": " + x.toString)}

    new StreamingKMeansModel(centers, counts)

  }

  /**
   * Main streaming operation: initialize the KMeans model
   * and then update it based on new data from the stream.
   */
  def runStreaming(data: DStream[Vector]): DStream[Int] = {
    var model = initRandom()
    data.foreachRDD(RDD => model = update(RDD, model))
    data.map(point => model.predict(point))
  }

}

/** Top-level methods for calling Streaming KMeans clustering. */
object StreamingKMeans {

  /**
   * Train a Streaming KMeans model. We initialize a set of
   * cluster centers randomly and then update them
   * after receiving each batch of data from the stream.
   * If a = 1 this is equivalent to sequential or mini-batch KMeans,
   * where each batch of data from the stream is treated as a different
   * mini-batch. If a < 1, perform forgetful KMeans, which
   * weights more recent data points more strongly.
   *
   * @param input Input DStream of (Array[Double]) data points
   * @param k Number of clusters to estimate.
   * @param d Number of dimensions per data point.
   * @param a Update rule (1 mini batch, < 1 forgetful).
   * @param maxIterations Maximum number of iterations per batch.
   * @param initializationMode Random initialization of cluster centers.
   * @return Output DStream of (Int) assignments of data points to clusters.
   */
  def trainStreaming(input: DStream[Vector],
      k: Int,
      d: Int,
      a: Double,
      maxIterations: Int,
      initializationMode: String)
    : DStream[Int] =
  {
    new StreamingKMeans().setK(k)
                         .setD(d)
                         .setAlpha(a)
                         .setMaxIterations(maxIterations)
                         .setInitializationMode(initializationMode)
                         .runStreaming(input)
  }

  def main(args: Array[String]) {
    if (args.length != 8) {
      System.err.println("Usage: StreamingKMeans <master> <directory> <batchTime> <k> <d> <a> <maxIterations> <initializationMode>")
      System.exit(1)
    }

    val (master, directory, batchTime, k, d, a, maxIterations, initializationMode) = (
      args(0), args(1), args(2).toLong, args(3).toInt, args(4).toInt, args(5).toDouble, args(6).toInt, args(7))

    val conf = new SparkConf().setMaster(master).setAppName("StreamingKMeans")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
        .setJars(List("target/scala-2.10/thunder_2.10-0.1.0.jar"))
        .set("spark.executor.memory", "100G")
    }

    /** Create Streaming Context */
    val ssc = new StreamingContext(conf, Seconds(batchTime))

    /** Train KMeans model */
    val data = LoadStreaming.fromText(ssc, directory).map(x => Vectors.dense(x))
    val assignments = StreamingKMeans.trainStreaming(data, k, d, a, maxIterations, initializationMode)

    /** Print assignments (for testing) */
    assignments.print()

    ssc.start()
    ssc.awaitTermination()
  }

}
