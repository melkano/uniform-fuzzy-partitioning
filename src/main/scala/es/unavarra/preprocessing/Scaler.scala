/*
 * Copyright (C) 2017 Mikel Elkano Ilintxeta and Mikel Galar Idoate
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package es.unavarra.preprocessing

import java.util.Arrays

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.mutable.HashSet

object Scaler {

  /**
    * Computes quantiles
    * @param dataset input dataset
    * @param variablesBounds variables' lower and upper bounds
    * @param numBins number of bins
    * @param sparkContext Spark context
    * @return quantiles' values
    */
  def computeQuantiles (dataset: RDD[String], variablesBounds: Array[(Double, Double, Boolean)], numBins: Int, sparkContext: SparkContext): Array[Array[Double]] = {
    val numSamples = dataset.count()
    val idx = for (i <- 1 to numBins) yield i
    val quantiles = Array(idx: _*).map(i => i / idx.length.toDouble)
    val computedQuantiles = new Array[Array[Double]](variablesBounds.length)
    var varIdx = 0
    while (varIdx < variablesBounds.length) {
      if (variablesBounds(varIdx)._3) {
        val sortedRdd = dataset.map(s => s.split(",")(varIdx).toDouble).sortBy(x => x, true).zipWithIndex()
        val quantilesIdx: HashSet[Long] = HashSet[Long]()
        var i = 0
        while (i < quantiles.length){
          quantilesIdx += Math.ceil(quantiles(i) * numSamples - 1).toLong.max(0L)
          i += 1
        }
        val quantilesBroad = sparkContext.broadcast(quantilesIdx)
        computedQuantiles(varIdx) = sortedRdd.filter(x => quantilesBroad.value.contains(x._2))
          .sortBy(x => x._2, true).map(x => x._1).collect()
      }
      varIdx += 1
    }
    computedQuantiles
  }

  /**
    * Creates a new dataset where all variables follow a uniform distribution
    * https://en.wikipedia.org/wiki/Probability_integral_transform
    * @param dataset dataset to be transformed
    * @param variablesBounds variables' lower and upper bounds
    * @param sparkContext Spark context
    * @param quantiles array containing the values of the quantiles
    * @return the transformed dataset where all the variables follow a uniform distribution along with the quantiles used for the transformation
    */
  def toUniform (dataset: RDD[String], variablesBounds: Array[(Double, Double, Boolean)], sparkContext: SparkContext, quantiles: Array[Array[Double]] = null): (RDD[String], Array[Array[Double]]) = {
    // Compute quantiles
    var quantilesBroad: Broadcast[Array[Array[Double]]] = quantiles match {
      case null => sparkContext.broadcast(computeQuantiles(dataset, variablesBounds, 1000, sparkContext)) // 1,000 bins are usually enough
      case other => sparkContext.broadcast(quantiles)
    }
    // Transform the dataset
    (dataset.map(s => {
      val inputVals = s.split(",")
      val outputVals: Array[String] = new Array[String](inputVals.length)
      var i = 0
      while (i < inputVals.length - 1){
        if (variablesBounds(i)._3) {
          val value = inputVals(i).toDouble
          // val quantileIdx = quantilesBroad.value(i).search(value).insertionPoint
          val index = Arrays.binarySearch(quantilesBroad.value(i), value)
          val quantileIdx = if (index < 0) Math.abs(index + 1) else index
          if (quantileIdx < quantilesBroad.value(i).length) {
            val x1 = quantileIdx match {
              case 0 => variablesBounds(i)._1
              case other => quantilesBroad.value(i)(quantileIdx - 1)
            }
            val x2 = quantilesBroad.value(i)(quantileIdx)
            val y1 = quantileIdx match {
              case 0 => 0d
              case other => quantileIdx / quantilesBroad.value(i).length.toDouble
            }
            val y2 = (quantileIdx + 1) / quantilesBroad.value(i).length.toDouble
            if ((x2 - x1) > 0) {
              val m = (y2 - y1) / (x2 - x1)
              outputVals(i) = (m * (value - x1) + y1).toString
            }
            else
              outputVals(i) = y1.toString
          }
          else
            outputVals(i) = "1.0"
        }
        else
          outputVals(i) = inputVals(i)
        i += 1
      }
      outputVals(outputVals.length - 1) = inputVals(inputVals.length - 1)
      outputVals.mkString(",")
    }), quantilesBroad.value)
  }

}
