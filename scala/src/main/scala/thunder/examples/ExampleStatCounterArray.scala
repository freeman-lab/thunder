package thunder.examples

import thunder.util.StatCounterArray

object ExampleStatCounterArray {

  def main(args: Array[String]) {
    val stats1 = StatCounterArray(Array(0.1, 0.2))
    val stats2 = StatCounterArray(Array(0.2, 0.3))
    val stats3 = StatCounterArray(Array(Array(0.1,0.2), Array[Double]()))
    println(stats1.merge(stats2).merge(stats3))

  }

}
