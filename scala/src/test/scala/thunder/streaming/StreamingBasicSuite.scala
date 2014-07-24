package thunder.streaming

import org.apache.spark.streaming.{Milliseconds, Seconds, StreamingContext}
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.util.StatCounter
import org.apache.spark.mllib.util.LinearDataGenerator
import cern.colt.matrix.{DoubleFactory1D, DoubleFactory2D}

import org.scalatest.FunSuite
import com.google.common.io.Files
import thunder.util.LoadStreaming
import scala.util.Random
import java.io.File
import org.apache.commons.io.FileUtils


/**
  * NOTE: Currently performing all streaming related tests
  * in one suite. I tried moving these exact same tests into separate
  * test suites but it caused several File IO related bugs,
  * still need to track them down.
  */
class StreamingBasicSuite extends FunSuite {

  import thunder.TestUtils._

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("test")

  test("streaming stats") {

    // set parameters
    val n = 1000 // number of data points per batch
    val m = 10 // number of batches
    val r = 0.05 // noise
    val mean = 3.0 // intercept for linear model
    val variance = 2.0 // coefficients for linear model

    // create test directory and set up streaming data
    val testDir = Files.createTempDir()
    val checkpointDir = Files.createTempDir()
    val ssc = new StreamingContext(conf, Seconds(1))
    val data = LoadStreaming.fromTextWithKeys(ssc, testDir.toString)

    // create and train linear model
    val state = StatefulStats.trainStreaming(data)
    var counter = new StatCounter()
    state.foreachRDD{rdd => counter = rdd.values.first()}
    ssc.checkpoint(checkpointDir.toString)
    ssc.start()

    val rand = new Random(41)

    Thread.sleep(5000)
    for (i <- 0 until m) {
      val samples = Array.fill(n)(rand.nextGaussian()).map(x => x * math.sqrt(variance) + mean)
      val file = new File(testDir, i.toString)
      FileUtils.writeStringToFile(file, "0 " + samples.mkString(" ") + "\n")
      Thread.sleep(Milliseconds(500).milliseconds)
    }
    Thread.sleep(Milliseconds(5000).milliseconds)

    ssc.stop()
    System.clearProperty("spark.driver.port")

    FileUtils.deleteDirectory(testDir)

    // compare estimated parameters to actual
    assertEqual(counter.mean, mean, 0.1)
    assertEqual(counter.variance, variance, 0.1)
    assertEqual(counter.count, n * m, 0.0001)

  }


  test("stateful linear regression") {

    // set parameters
    val n = 10 // number of data points per record
    val r = 0.05 // noise
    val m = 10 // number of batches
    val intercept = 1.0 // intercept for linear model
    val weights = 1.5 // coefficients for linear model

    // create test directory and set up streaming data
    val testDir = Files.createTempDir()
    val checkpointDir = Files.createTempDir()
    val ssc = new StreamingContext(conf, Seconds(1))
    val data = LoadStreaming.fromTextWithKeys(ssc, testDir.toString)

    // create and train stateful linear regression model
    val state = StatefulLinearRegression.trainStreaming(data, Array(0))
    var model = new FittedModel(0.0, 0.0, 0.0, 0.0, DoubleFactory2D.dense.make(0, 0), DoubleFactory1D.dense.make(0), DoubleFactory1D.dense.make(0))
    state.foreachRDD{rdd => model = rdd.values.first()}
    ssc.checkpoint(checkpointDir.toString)

    ssc.start()

    // generate streaming data
    val rand = new Random(41)

    Thread.sleep(5000)
    for (i <- 0 until m) {
      val feature = Array.fill(n)(rand.nextGaussian())
      val record = feature.map(x => x * weights + intercept + rand.nextGaussian() * r)
      val file = new File(testDir, i.toString)
      FileUtils.writeStringToFile(file, "0 " + feature.mkString(" ") + "\n" + "1 " + record.mkString(" ") + "\n")
      Thread.sleep(Milliseconds(500).milliseconds)
    }
    Thread.sleep(Milliseconds(5000).milliseconds)

    ssc.stop()
    System.clearProperty("spark.driver.port")

    FileUtils.deleteDirectory(testDir)
    FileUtils.deleteDirectory(checkpointDir)

    // compare estimated parameters to actual
    assertEqual(model.R2, 0.99, 0.1)
    assertEqual(model.intercept, intercept, 0.1)
    assertEqual(model.weights, Array(weights), 0.1)
    assertEqual(model.count, 100, 0.001)

  }

  test("streaming k means with single cluster") {

    // set parameters
    val k = 1 // number of clusters
    val d = 5 // number of dimensions
    val n = 50 // number of data points per batch
    val m = 10 // number of batches
    val r = 0.05 // noise

    // create test directory and set up streaming data
    val testDir = Files.createTempDir()
    val ssc = new StreamingContext(conf, Seconds(1))
    val data = LoadStreaming.fromText(ssc, testDir.toString).map(x => Vectors.dense(x))

    // create and train KMeans model
    val KMeans = new StreamingKMeans().setK(k).setD(d).setAlpha(1).setInitializationMode("gauss").setMaxIterations(1)
    var model = KMeans.initRandom()
    data.foreachRDD(RDD => model = KMeans.update(RDD, model))
    ssc.start()

    // generate streaming data
    val rand = new Random(42)
    val centers = Array.fill(k)(Array.fill(d)(rand.nextGaussian()))

    Thread.sleep(5000)
    for (i <- 0 until m) {
      val samples = Array.tabulate(n)(i => Array.tabulate(d)(i => centers(i % k)(i) + rand.nextGaussian() * r).mkString(" "))
      val file = new File(testDir, i.toString)
      FileUtils.writeStringToFile(file, samples.mkString("\n") + "\n")
      Thread.sleep(Milliseconds(500).milliseconds)
    }
    Thread.sleep(Milliseconds(5000).milliseconds)

    ssc.stop()
    System.clearProperty("spark.driver.port")

    FileUtils.deleteDirectory(testDir)

    // compare estimated center to actual
    assertSetsEqual(model.clusterCenters.map(x => x.toArray), centers, 0.1)
  }

  test("streaming linear regression") {

    // set parameters
    val n = 100 // number of data points per batch
    val m = 10 // number of batches
    val r = 0.05 // noise
    val intercept = 3.0 // intercept for linear model
    val weights = Array(2.0, 5.0) // coefficients for linear model

    // create test directory and set up streaming data
    val testDir = Files.createTempDir()
    val ssc = new StreamingContext(conf, Seconds(1))
    val data = LoadStreaming.fromTextWithLabels(ssc, testDir.toString)

    // create and train linear model
    val LinearModel = new StreamingLinearRegression(2, 1, 5, "fixed")
    var model = LinearModel.initFixed()
    data.foreachRDD{RDD => model = LinearModel.update(RDD, model)}
    ssc.start()

    Thread.sleep(5000)
    for (i <- 0 until m) {
      val samples = LinearDataGenerator.generateLinearInput(intercept, weights, n, 42*(i+1), r)
      val file = new File(testDir, i.toString)
      FileUtils.writeStringToFile(file, samples.map(x => x.label.toString + ", " + x.features.toArray.mkString(" ")).mkString("\n"))
      Thread.sleep(Milliseconds(500).milliseconds)
    }
    Thread.sleep(Milliseconds(5000).milliseconds)

    ssc.stop()
    System.clearProperty("spark.driver.port")

    FileUtils.deleteDirectory(testDir)

    // compare estimated parameters to actual
    assertEqual(model.intercept, intercept, 0.1)
    assertEqual(model.weights.toArray, weights, 0.1)

  }


}