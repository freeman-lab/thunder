package thunder.examples

import thunder.util.LoadParam

object ExampleLoadParam {

  def main(args: Array[String]) {

    val file = args(0)

    val param = LoadParam.fromText(file)

    println(param.toDebugString)

  }
}
