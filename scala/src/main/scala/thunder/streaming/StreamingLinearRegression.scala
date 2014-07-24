package thunder.streaming

import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.mllib.linalg.Vectors

import thunder.util.LoadStreaming
import scala.util.Random._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.streaming.StreamingLinearRegressionWithSGD

/**
 * Linear Regression on streaming data.
 *
 * The underlying assumption is that every data point
 * in the stream is a random label-feature pair, and
 * there is a single set of weights and coefficients
 * (the "linear model") that can predict each label
 * given the features. Given this assumption,
 * all streaming data points MUST the same
 * number of features.
 *
 * We update the weights by performing several
 * iterations of gradient descent on each batch
 * of streaming data. The number of data points
 * per batch can be arbitrary. Compared to single
 * batch algorithms, it should be OK to use
 * fewer iterations of gradient descent because
 * new updates will be performed for each batch.
 *
 * See also: StatefulLinearRegression
 */

class StreamingLinearRegression (
  var d: Int,
  var stepSize: Double,
  var numIterations: Int,
  var initializationMode: String) {

  /** Construct a StreamingLinearRegression object with default parameters */
  def this() = this(5, 1.0, 10, "fixed")

  val algorithm = new StreamingLinearRegressionWithSGD(stepSize, numIterations).setIntercept(addIntercept = true)

  /** Set the number of features per data point (d). Default: 5
    * TODO: if possible, set this automatically based on first data point
    */
  def setD(d: Int): StreamingLinearRegression = {
    this.d = d
    this
  }

  /**
   * Set the initialization mode, either random (gaussian) or fixed.
   * Default: fixed
   */
  def setInitializationMode(initializationMode: String): StreamingLinearRegression = {
    if (initializationMode != "random" && initializationMode != "fixed") {
      throw new IllegalArgumentException("Invalid initialization mode: " + initializationMode)
    }
    this.initializationMode = initializationMode
    this
  }

  /** Initialize a Linear Regression model with fixed weights */
  def initFixed(): LinearRegressionModel = {
    val weights = Vectors.dense(Array.fill(d)(1.0))
    val intercept = 0.0
    algorithm.createModel(weights, intercept)
  }

  /** Initialize a Linear Regression model with random weights */
  def initRandom(): LinearRegressionModel = {
    val weights = Vectors.dense(Array.fill(d)(nextGaussian()))
    val intercept = nextGaussian()
    algorithm.createModel(weights, intercept)
  }

  /** Update a Linear Regression model by running a gradient update */
  def update(rdd: RDD[LabeledPoint], model: LinearRegressionModel): LinearRegressionModel = {
    if (rdd.count() != 0) {
      algorithm.run(rdd, model.weights)
    } else {
      model
    }
  }

  /** Main streaming operation: initialize the Linear Regression model
    * and then update it based on new data from the stream.
    */
  def runStreaming(data: DStream[LabeledPoint]): DStream[Double] = {
    var model = initFixed()
    data.foreachRDD{rdd => model = update(rdd, model)}
    data.map(x => model.predict(x.features))
  }
}


/** Top-level methods for calling Streaming Linear Regression.*/
object StreamingLinearRegression {

  /**
   * Train a Streaming Linear Regression model. We initialize a model
   * and then perform gradient descent updates on each batch of
   * received data in the data stream (akin to mini-batch gradient descent
   * where each new batch from the stream is a different mini-batch).
   *
   * @param input Input DStream of (label, features) pairs.
   * @param d Number of features per data point.
   * @param stepSize Step size to be used for each iteration of Gradient Descent.
   * @param numIterations Number of iterations of gradient descent to run per batch.
   * @param initializationMode How to initialize model parameters (random or fixed).
   * @return DStream of (double) model predictions for data points.
   */
  def trainStreaming(
      input: DStream[LabeledPoint],
      d: Int,
      stepSize: Double,
      numIterations: Int,
      initializationMode: String)
  : DStream[Double] =
  {
    new StreamingLinearRegression(d, stepSize, numIterations, initializationMode).runStreaming(input)
  }

  def main(args: Array[String]) {
    if (args.length != 7) {
      System.err.println("Usage: StreamingLinearRegression <master> <directory> <batchTime> <d> <stepSize> <numIterations> <initializationMode>")
      System.exit(1)
    }

    val (master, directory, batchTime, d, stepSize, numIterations, initializationMode) = (
      args(0), args(1), args(2).toLong, args(3).toInt, args(4).toDouble, args(5).toInt, args(6).toString)

    val conf = new SparkConf().setMaster(master).setAppName("StreamingLinearRegression")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
        .setJars(List("target/scala-2.10/thunder_2.10-0.1.0.jar"))
        .set("spark.executor.memory", "100G")
    }

    /** Create Streaming Context */
    val ssc = new StreamingContext(conf, Seconds(batchTime))

    /** Train Streaming Linear Regression model */
    val data = LoadStreaming.fromTextWithLabels(ssc, directory)
    val predictions = StreamingLinearRegression.trainStreaming(data, d, stepSize, numIterations, initializationMode)

    /** Print predictions (for testing) */
    predictions.print()

    ssc.start()
    ssc.awaitTermination()
  }

}

