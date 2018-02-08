package iet.unipi.bigdatamining.classification.tree

import java.io.PrintWriter

import es.unavarra.preprocessing.Scaler
import iet.unipi.bigdatamining.classification.tree.model.FuzzyDecisionTreeModel
import iet.unipi.bigdatamining.discretization.FuzzyPartitioning
import iet.unipi.bigdatamining.discretization.model.FuzzyPartitioningModel
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.io.Source

object runFDT {

  val NUM_ARGS = 8

  def main(args: Array[String]): Unit = {

    if (args.length < NUM_ARGS) {
      System.err.println("Usage: <max_depth> <hdfs://url:port> <header_file> <train_file> <test_file> <num_fuzzy_sets> <num_partitions> <output_path>")
      System.exit(-1)
    }

    // Create Hadoop configuration
    val hadoopConf = new Configuration()

    // Parse arguments
    val maxDepth = args(0).toInt
    val hdfsLocation = args(1)
    val headerFile = args(2)
    val trainInputPath = hdfsLocation + "/" + args(3)
    val testInputPath = hdfsLocation + "/" + args(4)
    val numFS = args(5).toInt
    val numPartitions = args(6).toInt
    val outputPath = hdfsLocation + "/" + args(7)
    val outputDir = new Path(outputPath)
    val fs = FileSystem.get(hadoopConf)
    if (fs.exists(outputDir))
      fs.delete(outputDir, true)

    // Create Spark context
    // Logger.getLogger("org").setLevel(Level.OFF);
    // Logger.getLogger("akka").setLevel(Level.OFF);
    val conf = new SparkConf()
    //conf.set("spark.eventLog.enabled", "true")
    //conf.set("spark.eventLog.dir", parameters.outputPath)
    conf.set("spark.default.parallelism", numPartitions.toString) // https://spark.apache.org/docs/latest/tuning.html#level-of-parallelism
    print("Creating Spark context...  ")
    val sc: SparkContext = new SparkContext(conf)
    println("Done.\n")

    // Get categorical variables' values from header file
    val (classLabels, categoricalFeaturesValues, variablesBounds, numFeatures) = readHeaderFile(headerFile, hadoopConf)
    val numClasses = classLabels.length

    ///////////////////////////////////////////////////////////////////////////////////////
    //     Pre-processing: transform the original distribution into a uniform distribution
    ///////////////////////////////////////////////////////////////////////////////////////
    val tFP0 = System.currentTimeMillis()
    // Load data
    print("Transforming the original distribution into a uniform distribution...  ")
    // Transform the original distribution into a uniform one
    val (uniformTra, quantiles) = Scaler.toUniform(sc.textFile(trainInputPath, numPartitions).cache, variablesBounds, sc)
    println("Done.\n")
    // Transform data into an RDD of LabeledPoint
    val trainingData = toLabeledPoints(uniformTra, classLabels, categoricalFeaturesValues).cache

    ////////////////////////////
    //     Uniform partitioning
    ///////////////////////////
    val distanceFS = 1d / (numFS - 1) // distance between two consecutive cores
    val cores = Array.fill(numFS){1d}.zipWithIndex.map{case (value, idx) => 0 + idx * distanceFS}
    var thresholdsFeatureInfo: Map[Int, Array[Double]] = Map[Int, Array[Double]]()
    var i = 0
    while (i < numFeatures){
      if (variablesBounds(i)._3)  // Check whether it is a numeric feature
        thresholdsFeatureInfo += (i -> cores)
      i += 1
    }
    val fpModel = new FuzzyPartitioningModel(thresholdsFeatureInfo.map(f => (f._1, f._2.toList)))
    val tFP = System.currentTimeMillis() - tFP0

    //////////////////////////////////
    //     Fuzzy Decision Tree
    //////////////////////////////////

    // Remove features with less than two values
    val discardedFeatures = categoricalFeaturesValues.filter(f => f._2.length < 2).keySet
    val newIndices = Map(((for (i <- 0 until numFeatures) yield i).toSet -- discardedFeatures).toArray.sorted.zipWithIndex: _*)
    thresholdsFeatureInfo = (thresholdsFeatureInfo -- discardedFeatures).map(c => (newIndices(c._1), c._2))
    val categoricalFeaturesInfo = Map() ++ (categoricalFeaturesValues -- discardedFeatures).map(c => (newIndices(c._1), c._2.length))
    val trainingDataFiltered = discardFeaturesFromDataset(trainingData, discardedFeatures).cache()

    // Run the FDT algorihtm
    //   Empty categoricalFeaturesInfo indicates all features are continuous.
    //   Empty thresholdsFeatureInfo indicates all features are categorical.
    val impurity = "fuzzy_entropy"
    val tNorm = "Product"
    var maxBins = 32
    if (!categoricalFeaturesInfo.isEmpty)
      maxBins = categoricalFeaturesInfo.values.max.max(maxBins) // Increase the number of bins if a feature has more than 32 values
    if (maxBins > 32)
      println("\nSetting maximum number of bins to " + maxBins)

    // Train a FuzzyDecisionTreeModel model for classification.
    val t0Train = System.currentTimeMillis()
    print("\nTraining...  ")
    val fdtModel = FuzzyMultiDecisionTree.train(trainingDataFiltered,
      impurity, tNorm, maxDepth, maxBins, numClasses,
      categoricalFeaturesInfo, thresholdsFeatureInfo)
    val t1Train = System.currentTimeMillis()
    val tTrain = t1Train-t0Train
    println("Done ("+(tTrain/1000)+" seconds).")

    // Evaluate model on test instances and compute test error
    val uniformTst = Scaler.toUniform(sc.textFile(testInputPath, numPartitions).cache, variablesBounds, sc, quantiles)._1
    val testData = toLabeledPoints(uniformTst, classLabels, categoricalFeaturesValues).cache
    val testDataFiltered = discardFeaturesFromDataset(testData, discardedFeatures).cache()
    val labelAndPreds = testDataFiltered.map { point =>
      val prediction = fdtModel.predictByMaxValue(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testDataFiltered.count()

    // Build confusion matrix
    val confusionMatrixMap = labelAndPreds.map(p => (p, 1L)).reduceByKey(_+_).collect().toMap
    val confusionMatrix: Array[Array[Long]] = Array.fill(numClasses){Array.fill(numClasses){0L}}
    i = 0
    var j = 0
    while (i < numClasses){
      j = 0
      while (j < numClasses){
        confusionMatrix(i)(j) = confusionMatrixMap.getOrElse((i, j), 0L)
        j += 1
      }
      i += 1
    }
    val stats = computeAccuracyStats(confusionMatrix)

    // Print FDT Model complexity
    println(s"#Node: ${fdtModel.numNodes}")
    println(s"#Leaves: ${fdtModel.numLeaves}")
    println(s"#MaxDepth: ${fdtModel.depth}")
    println(s"#MinDepth: ${fdtModel.minDepth}")
    println(s"#AvgDepth: ${fdtModel.averageDepth}")

    // Print accuracy and model
    println("\nAccuracy rate = " + stats(0))
    println("Geometric mean = " + stats(1))
    println("Avg. accuracy per class = " + stats(2))

    // Save model, confusion matrix, and time
    saveModel(fpModel, fdtModel, outputPath + "/RB_stats.txt", hadoopConf)
    saveConfusionMatrix(outputPath, confusionMatrix, hadoopConf)
    writeElapsedTime(Array(tFP, tTrain, tFP+tTrain), Array("Time partitioning", "Time training", "Total time"), outputPath + "/Time.txt", hadoopConf)

  }

  /**
    * Returns an array of doubles containing the accuracy rate, geometric mean, and average accuracy per class
    * @return an array of doubles containing the accuracy rate, geometric mean, and average accuracy per class
    */
  def computeAccuracyStats(confusionMatrix: Array[Array[Long]]): Array[Float] = {
    val nClasses = confusionMatrix.length
    // Compute stats
    val hits: Array[Long] = Array.fill(nClasses){0}
    val nExamples: Array[Long] = Array.fill(nClasses){0}
    val TPrates: Array[Float] = Array.fill(nClasses){0}
    var gm: Float = 1
    var avgacc: Float = 0
    var i: Int = 0
    while (i < nClasses) {
      hits(i) = confusionMatrix(i)(i)
      nExamples(i) = confusionMatrix(i).sum
      TPrates(i) = hits(i).toFloat / nExamples(i).toFloat
      gm *= TPrates(i)
      avgacc += TPrates(i)
      i += 1
    }
    avgacc /= nClasses.toFloat
    gm = math.pow(gm, 1f / nClasses).toFloat
    val acc: Float = hits.sum.toFloat / nExamples.sum.toFloat
    Array(acc, gm, avgacc)
  }

  def discardFeaturesFromDataset(dataset: RDD[LabeledPoint], featuresToRemove: scala.collection.Set[Int]): RDD[LabeledPoint] = {
    dataset.map(p => {
      var selectedFeatures: Array[Double] = Array()
      var i: Int = 0
      while (i < p.features.size){
        if (!featuresToRemove.contains(i))
          selectedFeatures = selectedFeatures :+ p.features(i)
        i += 1
      }
      new LabeledPoint(p.label, Vectors.dense(selectedFeatures))
    })
  }

  /**
    * Reads categorical variables' values (label and features) and computes the total number of features (numeric and categorical)
    * @param filePath header file
    * @param hadoopConf Hadoop configuration
    * @return Categorical variables' values, variables' bounds, number of features: (label_values, Map[feature_index -> feature_values), variables_bounds, total_num_features)
    */
  def readHeaderFile(filePath: String, hadoopConf: org.apache.hadoop.conf.Configuration):
  (Array[String], Map[Int, Array[String]], Array[(Double, Double, Boolean)], Int) = {
    val fileContents = Source.fromInputStream(FileSystem.get(hadoopConf).open(
      new Path(filePath)).getWrappedStream).getLines.toList
    var outputVar: String = null
    var varValues: Map[String, (Int, Array[String])] = Map[String, (Int, Array[String])]()
    var variablesBounds: Array[(Double, Double, Boolean)] = Array[(Double, Double, Boolean)]()
    var varIdx: Int = 0
    try {
      fileContents.foreach { str =>
        str match {
          case str if str.contains("@attribute") => {
            val attributePattern = """(@attribute)[ ]+([\w\-/]+)[ ]*(integer|real)?[ ]*[\[{]([^{}\[\]]*)[\]}]""".r // \w, &+:;/()\.\-
            val regexp = attributePattern.findFirstMatchIn(str).get
            val attName = regexp.group(2)
            val attType = regexp.group(3)
            var attRangeOriginal = regexp.group(4).split(",").map(_.trim)
            attType match {
              case "real" | "integer" => {
                val attRange: (Double, Double) = (attRangeOriginal(0).toDouble, attRangeOriginal(1).toDouble)
                variablesBounds = variablesBounds :+ (attRange._1, attRange._2, true)
              }
              case _ => {
                varValues += (attName -> (varIdx, attRangeOriginal))
                variablesBounds = variablesBounds :+ (0d, 0d, false)
              }
            }
            varIdx += 1
          }
          case str if str.contains("@outputs") => outputVar = str.substring("@outputs".length).replaceAll(" ", "")
          case _ =>
        }
      }
    } catch {
      case e: Exception =>
        System.err.println("\nERROR READING HEADER FILE: Error while parsing @ fields \n")
        System.err.println(e)
        System.exit(-1)
    }
    (varValues(outputVar)._2, (varValues - outputVar).map(v => v._2), variablesBounds, varIdx - 1)
  }

  /**
    * Save the confusion matrix and accuracy stats to disk
    */
  def saveConfusionMatrix(outputPath: String, confusionMatrix: Array[Array[Long]],
                          hadoopConf: org.apache.hadoop.conf.Configuration): Unit = {
    var pw = new PrintWriter(FileSystem.get(hadoopConf).create(
      new Path(outputPath + "/ConfusionMatrix.txt"),true).getWrappedStream
    )
    pw.write(confusionMatrix.map(_.mkString("\t")).mkString("\n"))
    pw.close()
    // Save stats
    pw = new PrintWriter(FileSystem.get(hadoopConf).create(
      new Path(outputPath + "/Accuracy.txt"),true).getWrappedStream
    )
    val stats = computeAccuracyStats(confusionMatrix)
    pw.write("Accuracy rate = " + stats(0) + "\n")
    pw.write("Geometric mean = " + stats(1) + "\n")
    pw.write("Avg. accuracy per class = " + stats(2) + "\n")
    pw.close()
  }

  def saveModel(fpModel: FuzzyPartitioningModel, fdtModel: FuzzyDecisionTreeModel,
                fileName: String, hadoopConf: org.apache.hadoop.conf.Configuration): Unit = {
    var pw = new PrintWriter(FileSystem.get(hadoopConf).create(
      new Path(fileName),true).getWrappedStream
    )
    pw.write("Totoal number of Fuzzy Sets: "+fpModel.numFuzzySets+"\n")
    pw.write("Average number of Fuzzy Sets: "+fpModel.averageFuzzySets+"\n")
    pw.write("#Discarded features: "+fpModel.discardedFeature+"\n")
    pw.write("#Fuzzy sets of the feature with the highest number of fuzzy sets: "+fpModel.max._2+"\n")
    pw.write("#Fuzzy sets of the feature with the lowest number of fuzzy sets (discarded features are not taken in considiration): "+fpModel.min._2+"\n")
    pw.write("#Node: "+fdtModel.numNodes+"\n")
    pw.write("#Leaves: "+fdtModel.numLeaves+"\n")
    pw.write("#MaxDepth: "+fdtModel.depth+"\n")
    pw.write("#MinDepth: "+fdtModel.minDepth+"\n")
    pw.write("#AvgDepth: "+fdtModel.averageDepth+"\n")
    pw.close()
  }

  def toLabeledPoints(dataset: RDD[String], classLabels: Array[String], categoricalValues: Map[Int, Array[String]]): RDD[LabeledPoint] = {
    dataset.map(s => {
      // Assume class label is the last item
      val example = s.split(",")
      val label = example.last
      val features = example.dropRight(1)
      categoricalValues.foreach(v => features(v._1) = v._2.indexOf(features(v._1)).toString)
      new LabeledPoint(classLabels.indexOf(label), Vectors.dense(features.map(c => c.toDouble)))
    })
  }

  def writeElapsedTime(elapsedTimes: Array[Long], titles: Array[String], fileName: String, hadoopConf: org.apache.hadoop.conf.Configuration): String = {
    val pw = new PrintWriter(FileSystem.get(hadoopConf).create(
      new Path(fileName),true).getWrappedStream
    )
    var str = ""
    var i: Int = 0
    while (i < elapsedTimes.length) {
      val h = elapsedTimes(i) / 3600000;
      val m = (elapsedTimes(i) % 3600000) / 60000;
      val s = ((elapsedTimes(i) % 3600000) % 60000) / 1000;
      str += f"${titles(i)} (hh:mm:ss): $h%02d:$m%02d:$s%02d (${elapsedTimes(i) / 1000} seconds)\n"
      i += 1
    }
    pw.write(str)
    pw.close()
    str
  }

}
