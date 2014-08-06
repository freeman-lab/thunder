package thunder.examples

import org.apache.spark.{SparkContext, SparkConf}
import thunder.util.Load

object ExampleLoad {

  def main(args: Array[String]) {

    val master = args(0)

    val file = args(1)

    val conf = new SparkConf().setMaster(master).setAppName("ExampleLoad")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
        .setJars(List("target/scala-2.10/thunder_2.10-0.1.0.jar"))
        .set("spark.executor.memory", "100G")
    }

    val sc = new SparkContext(conf)

    val data = Load.fromBinaryWithKeys(sc, file, nKeys = 1)

    data.take(10).foreach{x =>
      println(x._1.mkString(" "))
      println(x._2.mkString(" "))
    }

    println(data.count())

  }
}
