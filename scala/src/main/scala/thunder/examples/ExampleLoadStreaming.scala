package thunder.examples

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import thunder.util.LoadStreaming

object ExampleLoadStreaming {

  def main(args: Array[String]) {

    val master = args(0)

    val file = args(1)

    val batchTime = args(2).toLong

    val conf = new SparkConf().setMaster(master).setAppName("ExampleLoadStreaming")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
        .setJars(List("target/scala-2.10/thunder_2.10-0.1.0.jar"))
        .set("spark.executor.memory", "100G")
    }

    val ssc = new StreamingContext(conf, Seconds(batchTime))

    val data = LoadStreaming.fromBinary(ssc, file)

    data.map(v => v.mkString(" ")).print()

    data.foreachRDD(rdd => println(rdd.count()))

    ssc.start()
    ssc.awaitTermination()

  }
}
