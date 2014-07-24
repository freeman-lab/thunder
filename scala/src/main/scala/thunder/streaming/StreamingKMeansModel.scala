package org.apache.spark.mllib.streaming

import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vector

/**
 * Extends clustering model for K-means with the current counts of each cluster (for streaming algorithms)
 */
class StreamingKMeansModel(override val clusterCenters: Array[Vector],
                           val clusterCounts: Array[Int] = Array(1)) extends KMeansModel(clusterCenters)
