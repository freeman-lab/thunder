package org.apache.spark.mllib.streaming

import org.apache.spark.mllib.regression.{LinearRegressionModel, GeneralizedLinearAlgorithm}
import org.apache.spark.mllib.optimization.{GradientDescent, SimpleUpdater, LeastSquaresGradient}
import org.apache.spark.mllib.linalg.Vector

/**
 * Duplicate of LinearRegressionWithSGD but with public methods so the companion
 * streaming algorithms can call the underlying methods
 */
class StreamingLinearRegressionWithSGD (
  var stepSize: Double,
  var numIterations: Int) extends GeneralizedLinearAlgorithm[LinearRegressionModel] with Serializable {

  val gradient = new LeastSquaresGradient()
  val updater = new SimpleUpdater()
  val optimizer =  new GradientDescent(gradient, updater).setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setMiniBatchFraction(1.0)

  /** Create a Linear Regression model */
  def createModel(weights: Vector, intercept: Double) = {
    new LinearRegressionModel(weights, intercept)
  }

}