package thunder.util.io

/**
 * Class for pre processing data
 *
 * @param preProcessMethod Method of pre processing
 */
case class PreProcessor(preProcessMethod: String) {
  def get(x: Array[Double]): Array[Double] = preProcessMethod match {

    case "raw" => x

    case "meanSubtract" =>
      val mean = x.sum / x.length
      x.map(_-mean)

    case "dff" =>
      val mean = x.sum / x.length
      x.map{ x =>
        (x - mean) / (mean + 0.1) }
  }
}

