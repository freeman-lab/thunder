
package thunder.examples

import org.apache.spark.{SparkContext, SparkConf}
import thunder.util.Save

object ExampleSave {

  def main(args: Array[String]) {

    val master = args(0)

    val file = args(1)

    val directory = args(2)

    val conf = new SparkConf().setMaster(master).setAppName("ExampleLoad")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
        .setJars(List("target/scala-2.10/thunder_2.10-0.1.0.jar"))
        .set("spark.executor.memory", "100G")
    }

    val sc = new SparkContext(conf)

    val data = sc.parallelize(Seq((Array(3),Array(1.5)),(Array(2),Array(2.0)),(Array(1),Array(3.0))))

    Save.asTextWithKeys(data, directory, Seq(file))

  }
}
