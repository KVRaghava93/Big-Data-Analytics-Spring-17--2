import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

/**
  * Created by raghava koundinya on 2/8/2017.
  */
object ChimpKmeans {
  def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir","F:\\winutils");

    val sparkConf = new SparkConf().setAppName("ChimpDataKmeans").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);
    // Load and parse the data
    val chimpdata = sc.textFile("data/chimpanzeedata2.txt")

    val parsedData = chimpdata.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    //Look at how training data is!
    parsedData.foreach(f=>println(f))

    // Cluster the data into two classes using KMeans
    val numofClusters = 3
    val numofIterations = 15
    val clusters = KMeans.train(parsedData, numofClusters, numofIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val SumOfSquaredErrors = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + SumOfSquaredErrors)

    //Look at how the clusters are in training data by making predictions
    println("Clustering on training data: ")
    clusters.predict(parsedData).zip(parsedData).foreach(f=>println(f._2,f._1))

    // Save and load model
    clusters.save(sc, "data/ChimpDataKmeans")
    val sameModel = KMeansModel.load(sc, "data/ChimpDataKmeans")


  }




}
