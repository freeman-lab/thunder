package thunder

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.scalatest.BeforeAndAfterAll
import org.scalatest.Suite

/** Shares a local 'SparkContext' between all tests in a suite and closes it at the end.
  * (adapted from Spark testing suite) */
trait SharedSparkContext extends BeforeAndAfterAll { self: Suite =>

  @transient private var _sc: SparkContext = _

  def sc: SparkContext = _sc

  var conf = new SparkConf(false)

  override def beforeAll() {
    _sc = new SparkContext("local[2]", "test", conf)
    super.beforeAll()
  }

  override def afterAll() {
    LocalSparkContext.stop(_sc)
    _sc = null
    super.afterAll()
  }
}